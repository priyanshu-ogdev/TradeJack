"""
Risk guardian: the safety layer between a model's decision and the paper (later,
real) exchange. Every check here is deliberately hard-coded and dumb — it should
never require the model's cooperation to work, since the entire point is to
catch cases where the model, the feed, or the risk logic itself has a bug.

This runs in paper mode from day one (not bolted on only once real money
arrives) specifically so any bug in the guardian itself surfaces against fake
money first.
"""

import os
import json
import time
import logging
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import Optional, Any, Dict, List, Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] (RiskGuardian) %(message)s")
logger = logging.getLogger("RiskGuardian")


@dataclass
class RiskLimits:
    max_position_fraction: float = 0.5      # mirrors scripts/deploy_config.py DeploymentConfig
    min_hold_ticks: int = 100
    max_daily_loss_pct: float = 0.05
    max_drawdown_halt: float = 0.15
    max_orders_per_minute: int = 6
    max_book_staleness_sec: float = 2.0
    # NOTE: kill_switch_path/active_halt_path/alert_history_path below are bare
    # relative paths by default, resolving against the process's CURRENT
    # WORKING DIRECTORY at launch, not any shared "state directory" concept --
    # there is no resolution logic in this project tying them together. Every
    # real construction site MUST override all three consistently (see
    # execution/live_inference_server.py's limits_kwargs:
    # os.path.join(state_dir, "...") for each). Anchoring only kill_switch_path
    # while leaving the other two on their bare defaults was a real bug
    # (caught on review, not a live incident) that silently sent the halt
    # marker and alert history to a different directory than the rest of a
    # deployment's state whenever launched from an unexpected working
    # directory -- defeating the sticky-halt-plus-alerting design's entire
    # point without anything failing loudly.
    kill_switch_path: str = "state/KILL_SWITCH"   # touch this file to halt trading immediately
    # ── Halt alerting (added: was entirely missing -- a sticky halt with no
    # signal to anyone was functionally indistinguishable from a silent crash) ──
    active_halt_path: str = "state/ACTIVE_HALT.json"       # exists iff currently halted
    alert_history_path: str = "state/halt_alerts.log.jsonl"  # append-only, survives resolves
    halt_reminder_interval_sec: float = 300.0              # re-alert at most this often while halted


@dataclass
class RiskState:
    is_halted: bool = False
    halt_reason: str = ""
    current_equity: float = 100.0
    daily_start_equity: float = 100.0
    peak_equity: float = 100.0
    orders_last_minute: int = 0


class RiskGuardian:
    """
    Risk guardian safety layer. Supports both:
    1. Direct per-tick check(target_frac, ...) used by LivePaperInferenceServer
    2. Order-by-order check_order_allowed(side, qty, price) used by ExchangeAdapter/tests
    """

    def __init__(
        self,
        limits: Optional[RiskLimits] = None,
        exchange: Optional[Any] = None,
        symbol: str = "BTC/USDT",
        max_position_fraction: Optional[float] = None,
        max_daily_loss_pct: Optional[float] = None,
        max_drawdown_halt: Optional[float] = None,
        min_hold_ticks: Optional[int] = None,
        max_orders_per_minute: Optional[int] = None,
        starting_equity: Optional[float] = None,
        **kwargs,
    ):
        if limits is None:
            limits = RiskLimits(
                max_position_fraction=max_position_fraction if max_position_fraction is not None else 0.5,
                max_daily_loss_pct=max_daily_loss_pct if max_daily_loss_pct is not None else 0.05,
                max_drawdown_halt=max_drawdown_halt if max_drawdown_halt is not None else 0.15,
                min_hold_ticks=min_hold_ticks if min_hold_ticks is not None else 100,
                max_orders_per_minute=max_orders_per_minute if max_orders_per_minute is not None else 6,
            )
        self.limits = limits
        self.exchange = exchange
        self.symbol = symbol
        # BUG FIX: `starting_equity` used to default to a hard-coded 100.0.
        # Nothing at any real construction site (live_inference_server.py)
        # ever passed this argument, so RiskGuardian's daily-loss baseline
        # silently diverged from whatever the exchange/paper-wallet was
        # actually funded with. Concretely: LivePaperInferenceServer's own
        # CLI defaults `--cash` to $10 and DeployConfig.starting_capital
        # defaults to $100 -- either way, RiskGuardian's real construction
        # call site (`RiskGuardian(RiskLimits(**limits_kwargs))`) passed
        # neither, so it always silently assumed $100. Any account actually
        # funded below $100 (the $10 CLI default included) would compute a
        # fabricated daily loss on the very first check -- e.g.
        # (10 - 100) / 100 = -90%, tripping the 5% daily-loss halt before a
        # single order is ever placed. Reproduced directly by this file's
        # own __main__ self-test, which failed before this fix.
        # Fix: leave the baseline unset (None) unless a caller explicitly
        # knows the true starting equity up front. `_reset_daily_if_needed`
        # already has (and always had) the correct lazy-capture behavior for
        # this -- it sets the baseline from the first real `equity` it's
        # given -- it just never got the chance to run because this
        # constructor was pre-filling a wrong value first.
        self.starting_equity = starting_equity
        seed_equity = starting_equity if starting_equity is not None else 0.0
        self.state = RiskState(
            current_equity=seed_equity,
            daily_start_equity=seed_equity,
            peak_equity=seed_equity,
        )
        self._order_timestamps: list = []
        self._last_flip_tick: int = -10**9
        self._current_tick: int = 0
        self._daily_start_equity: Optional[float] = starting_equity
        self._daily_start_time = time.time()
        self._halted = False
        self._halt_reason: Optional[str] = None
        self._halt_triggered_at: Optional[float] = None
        self._last_alert_at: Optional[float] = None

    # ------------------------------------------------------------------ #
    # Halt + alerting -- single consolidated path
    # ------------------------------------------------------------------ #
    #
    # Design decision this implements (the operator's explicit choice, not a
    # default I picked): sticky halt + alerting, over auto-clearing at the UTC
    # daily boundary. A halt should be rare, loud, and end only when a human
    # decides it should -- auto-clearing trades that certainty away for uptime
    # nobody actually asked for if the underlying cause wasn't a one-off. The
    # real risk in "sticky, no alert" was never the stopping, it was that
    # nobody found out. This is the alerting half of that decision.

    def _trigger_halt(self, reason_code: str, message: str):
        """Single path for entering a halted state. This replaces what used to
        be six separate call sites (two in update_equity, one in
        check_order_allowed, three in check()), each independently setting
        self._halted / self.state.is_halted / self._halt_reason /
        self.state.halt_reason and logging critical by hand. Six copies of a
        four-line pattern is exactly the kind of thing that silently drifts --
        e.g. a copy that sets self._halted but not self.state.is_halted would
        leave check() refusing orders while check_order_allowed() still
        approved them, and nobody would notice until it mattered. One path,
        used everywhere a halt can start."""
        self._halted = True
        self.state.is_halted = True
        self._halt_reason = reason_code
        self.state.halt_reason = message
        self._halt_triggered_at = time.time()
        self._last_alert_at = self._halt_triggered_at
        logger.critical(f"RiskGuardian HALT [{reason_code}]: {message}")
        self._fire_alert("triggered", mark_active=True, reason_code=reason_code, message=message)

    def _maybe_send_reminder(self):
        """Called every time a check is made against an already-halted guardian
        -- specifically from *inside* the early-return branches of check() and
        check_order_allowed(), not after them. An earlier draft of this put the
        halted-early-return before anything that could reach a reminder call,
        which made the entire reminder path dead code on every call after the
        first halt -- caught by test_reminder_fires_while_halted below, which
        is the thing that actually proves this call site is reachable; if that
        test is ever deleted, this comment is the only record of why the call
        site inside the early return (not after it) matters.

        Re-fires at most once per halt_reminder_interval_sec, not on every
        call, so this doesn't become its own source of alert fatigue -- the
        point is to give an operator who missed the original alert a second
        (and third, and Nth) chance to notice, not to spam."""
        now = time.time()
        if self._last_alert_at is None or (now - self._last_alert_at) >= self.limits.halt_reminder_interval_sec:
            self._last_alert_at = now
            self._fire_alert("reminder", mark_active=True, reason_code=self._halt_reason, message=self.state.halt_reason)

    def _fire_alert(self, event: str, mark_active: bool, reason_code: Optional[str], message: Optional[str]):
        """Writes one entry to the append-only alert history, and separately
        maintains active_halt_path as a single current-state pointer file.

        The mark_active split exists because of a bug in an earlier version of
        this: it always (re)wrote the active-marker file regardless of which
        event fired, so calling it for a 'resolved' event -- right after
        reset_halt() had just deleted that same file -- immediately recreated
        it, leaving a misleading 'still halted' marker behind a halt that had
        actually just been cleared. Now: mark_active=True writes/refreshes the
        marker (triggered, reminder); mark_active=False only appends to
        history and removes the marker if present (resolved). The two
        branches can't step on each other because only one of them ever
        touches the marker file's existence."""
        ts = time.time()
        entry = {
            "event": event,  # "triggered" | "reminder" | "resolved"
            "reason_code": reason_code,
            "message": message,
            "timestamp": ts,
            "timestamp_iso": datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
        }
        try:
            hist_dir = os.path.dirname(self.limits.alert_history_path)
            if hist_dir:
                os.makedirs(hist_dir, exist_ok=True)
            with open(self.limits.alert_history_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
        except OSError as e:
            logger.error(f"RiskGuardian: failed to append alert history ({self.limits.alert_history_path}): {e}")

        try:
            if mark_active:
                active_dir = os.path.dirname(self.limits.active_halt_path)
                if active_dir:
                    os.makedirs(active_dir, exist_ok=True)
                with open(self.limits.active_halt_path, "w", encoding="utf-8") as f:
                    json.dump({
                        "reason_code": reason_code,
                        "message": message,
                        "halted_at": self._halt_triggered_at,
                        "halted_at_iso": (
                            datetime.fromtimestamp(self._halt_triggered_at, tz=timezone.utc).isoformat()
                            if self._halt_triggered_at else None
                        ),
                        "last_alert_at": ts,
                        "last_alert_at_iso": entry["timestamp_iso"],
                        "last_alert_event": event,
                    }, f, indent=2)
            else:
                if os.path.exists(self.limits.active_halt_path):
                    os.remove(self.limits.active_halt_path)
        except OSError as e:
            logger.error(f"RiskGuardian: failed to update active-halt marker ({self.limits.active_halt_path}): {e}")

    @staticmethod
    def read_active_halt(active_halt_path: str = "state/ACTIVE_HALT.json") -> Optional[dict]:
        """Reads the active-halt marker without needing a live RiskGuardian
        instance -- for a dashboard, supervisor, or alerting script to poll.
        Returns None if not currently halted (file absent) rather than raising,
        since 'not halted' is the normal, expected state, not an error."""
        if not os.path.exists(active_halt_path):
            return None
        try:
            with open(active_halt_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            logger.error(f"RiskGuardian: failed to read active-halt marker ({active_halt_path}): {e}")
            return None

    def _reset_daily_if_needed(self, equity: float):
        if self._daily_start_equity is None:
            self._daily_start_equity = equity
            self._daily_start_time = time.time()
            return
        if (time.time() - self._daily_start_time) >= 86400:
            self._daily_start_equity = equity
            self._daily_start_time = time.time()

    def kill_switch_engaged(self) -> bool:
        return os.path.exists(self.limits.kill_switch_path)

    def update_equity(self, current_equity: float):
        """Update tracked equity and verify daily loss / drawdown thresholds."""
        self.state.current_equity = current_equity
        if current_equity > self.state.peak_equity:
            self.state.peak_equity = current_equity
        self._reset_daily_if_needed(current_equity)

        if self._daily_start_equity and self._daily_start_equity > 0:
            loss_pct = (self._daily_start_equity - current_equity) / self._daily_start_equity
            if loss_pct >= self.limits.max_daily_loss_pct:
                self._trigger_halt("max_daily_loss_breached", f"Daily loss {loss_pct*100:.1f}% exceeded limit")

        if self.state.peak_equity > 0:
            dd = (self.state.peak_equity - current_equity) / self.state.peak_equity
            if dd >= self.limits.max_drawdown_halt:
                self._trigger_halt("max_drawdown_breached", f"Drawdown {dd*100:.1f}% exceeded limit")

    def check_order_allowed(self, side: str, qty: float, price: float) -> tuple:
        """Compatibility check: verifies order against position size, rate limits, and halts."""
        if self._halted or self.state.is_halted:
            self._maybe_send_reminder()
            return False, f"Trading halted: {self.state.halt_reason or self._halt_reason}"

        if self.kill_switch_engaged():
            self._trigger_halt("kill_switch_engaged", f"Kill switch file present at {self.limits.kill_switch_path}")
            return False, "KILL SWITCH ENGAGED"

        order_cost = abs(qty * price)
        max_cost = self.state.current_equity * self.limits.max_position_fraction
        if order_cost > max_cost + 1e-6:
            return False, f"Position size ${order_cost:.2f} exceeds limit ${max_cost:.2f}"

        now = time.time()
        self._order_timestamps = [t for t in self._order_timestamps if now - t < 60.0]
        if len(self._order_timestamps) >= self.limits.max_orders_per_minute:
            return False, f"Rate limit: {len(self._order_timestamps)} orders placed in the last 60s"

        return True, "PASSED"

    def record_order_executed(self, side: str, qty: float):
        """Record order execution timestamp for rate limiting."""
        self._order_timestamps.append(time.time())

    async def execute_safe_order(self, side: str, qty: float, price: float):
        """Pre-trade risk check followed by order execution on exchange if approved."""
        allowed, reason = self.check_order_allowed(side, qty, price)
        if not allowed:
            from execution.exchange_adapter import OrderResult
            return OrderResult(
                order_id="REJECTED-RISK",
                symbol=self.symbol,
                side=side,
                qty=0.0,
                avg_price=0.0,
                cost=0.0,
                fee=0.0,
                timestamp=time.time(),
                status="rejected",
                raw={"reason": reason},
            )
        if self.exchange is not None:
            from execution.exchange_adapter import ExchangeBannedError, OrderResult
            try:
                res = await self.exchange.place_market_order(self.symbol, side, qty)
                self.record_order_executed(side, qty)
                return res
            except ExchangeBannedError as e:
                # place_market_order raises this rather than returning a status
                # when already inside a known ban window (no request sent at
                # all) — surfacing it as a normal OrderResult here rather than
                # an uncaught exception, since a risk layer crashing is worse
                # than a risk layer reporting "banned" and letting the caller
                # decide how to halt.
                logger.critical(f"Exchange is banned, order not attempted: {e}")
                return OrderResult(
                    order_id="BANNED", symbol=self.symbol, side=side, qty=0.0, avg_price=0.0,
                    cost=0.0, fee=0.0, timestamp=time.time(), status="banned", raw={"error": str(e)},
                )
        return None

    def risk_budget_used_fraction(self) -> float:
        """Fraction (0.0-1.0+) of the daily loss budget consumed so far, clipped to
        [0, 1] -- for composition_layer.py's pre-emptive risk-budget throttle
        (see execution/composition_layer.py, docs/COMPOSITION_LAYER.md). Same formula
        already used internally by update_equity()/check() for the hard halt, exposed
        here as a single source of truth rather than a third duplicate of the formula.
        Returns 0.0 (no budget consumed) if daily_start_equity isn't set yet or is
        non-positive, or if today's equity is at/above the day's starting equity
        (a gain, not a loss, consumes none of the loss budget)."""
        if not self._daily_start_equity or self._daily_start_equity <= 0:
            return 0.0
        loss_pct = (self._daily_start_equity - self.state.current_equity) / self._daily_start_equity
        if loss_pct <= 0 or self.limits.max_daily_loss_pct <= 0:
            return 0.0
        return min(1.0, loss_pct / self.limits.max_daily_loss_pct)

    def get_risk_summary(self) -> dict:
        """Return risk guardian status summary."""
        now = time.time()
        self._order_timestamps = [t for t in self._order_timestamps if now - t < 60.0]
        is_conn = True
        if self.exchange is not None and hasattr(self.exchange, "is_connected"):
            is_conn = self.exchange.is_connected()
        return {
            "equity": self.state.current_equity,
            "is_halted": self.state.is_halted or self._halted,
            "halt_reason": self.state.halt_reason or self._halt_reason,
            "exchange_connected": is_conn,
            "orders_last_minute": len(self._order_timestamps),
            "active_alert": RiskGuardian.read_active_halt(self.limits.active_halt_path),
            "risk_budget_used_fraction": self.risk_budget_used_fraction(),
        }

    def check(
        self,
        target_frac: float,
        current_tick: int,
        equity: float,
        peak_equity: float,
        max_drawdown: float,
        book_is_stale: bool,
    ) -> tuple:
        """Returns (approved: bool, reason: Optional[str])."""
        if self._halted:
            self._maybe_send_reminder()
            return False, f"halted_previously:{self._halt_reason}"

        if self.kill_switch_engaged():
            self._trigger_halt("kill_switch_file_present", f"Kill switch file present at {self.limits.kill_switch_path}")
            return False, self._halt_reason

        if book_is_stale:
            return False, "book_stale"

        if abs(target_frac) > self.limits.max_position_fraction:
            return False, "exceeds_max_position_fraction"

        if (current_tick - self._last_flip_tick) < self.limits.min_hold_ticks:
            return False, "min_hold_ticks_not_elapsed"

        now = time.time()
        self._order_timestamps = [t for t in self._order_timestamps if now - t < 60.0]
        if len(self._order_timestamps) >= self.limits.max_orders_per_minute:
            return False, "rate_limited"

        self._reset_daily_if_needed(equity)
        if self._daily_start_equity and self._daily_start_equity > 0:
            daily_pnl_pct = (equity - self._daily_start_equity) / self._daily_start_equity
            if daily_pnl_pct <= -self.limits.max_daily_loss_pct:
                self._trigger_halt(
                    "max_daily_loss_breached",
                    f"MAX DAILY LOSS BREACHED: {daily_pnl_pct*100:.2f}% "
                    f"(limit {-self.limits.max_daily_loss_pct*100:.1f}%)",
                )
                return False, self._halt_reason

        if max_drawdown >= self.limits.max_drawdown_halt:
            self._trigger_halt(
                "max_drawdown_breached",
                f"MAX DRAWDOWN BREACHED: {max_drawdown*100:.2f}% "
                f"(limit {self.limits.max_drawdown_halt*100:.1f}%)",
            )
            return False, self._halt_reason

        return True, None

    def record_order_submitted(self, tick: int):
        self._order_timestamps.append(time.time())
        self._last_flip_tick = tick

    def reset_halt(self):
        """Manual override to resume after a halt -- not called automatically.
        This is the human-decision half of the sticky-halt-plus-alerting design:
        a resume only ever happens because this was called, never on a timer.
        Clears state, appends a 'resolved' entry to the alert history (a
        permanent record that this was deliberate), and removes the
        active-halt marker file."""
        logger.warning(f"Risk guardian halt manually reset (was: {self._halt_reason}).")
        self._fire_alert("resolved", mark_active=False, reason_code=self._halt_reason, message="Manually reset by operator")
        self._halted = False
        self.state.is_halted = False
        self._halt_reason = None
        self.state.halt_reason = ""
        self._halt_triggered_at = None
        self._last_alert_at = None


if __name__ == "__main__":
    print("Smoke-testing RiskGuardian...")
    guard = RiskGuardian(RiskLimits(max_daily_loss_pct=0.05, max_drawdown_halt=0.15, min_hold_ticks=5, kill_switch_path="/tmp/nonexistent_kill_switch"))

    ok, reason = guard.check(0.9, current_tick=1, equity=10.0, peak_equity=10.0, max_drawdown=0.0, book_is_stale=False)
    print("over-limit position ->", ok, reason)
    assert not ok and reason == "exceeds_max_position_fraction"

    ok, reason = guard.check(0.4, current_tick=1, equity=10.0, peak_equity=10.0, max_drawdown=0.0, book_is_stale=False)
    print("valid order ->", ok, reason)
    assert ok
    guard.record_order_submitted(tick=1)

    ok, reason = guard.check(0.4, current_tick=2, equity=10.0, peak_equity=10.0, max_drawdown=0.0, book_is_stale=False)
    print("too soon (min_hold_ticks) ->", ok, reason)
    assert not ok and reason == "min_hold_ticks_not_elapsed"

    ok, reason = guard.check(0.4, current_tick=10, equity=9.4, peak_equity=10.0, max_drawdown=0.06, book_is_stale=False)
    print("5%% daily loss breach ->", ok, reason)
    assert not ok and reason == "max_daily_loss_breached"

    guard2 = RiskGuardian(RiskLimits(min_hold_ticks=0, kill_switch_path="/tmp/nonexistent_kill_switch"))
    ok, reason = guard2.check(0.1, current_tick=1, equity=10.0, peak_equity=10.0, max_drawdown=0.0, book_is_stale=True)
    print("stale book ->", ok, reason)
    assert not ok and reason == "book_stale"

    print("All RiskGuardian assertions passed.")

    # ---------------------------------------------------------------- #
    # Alerting behavior -- executable, not just traced, since this file
    # has no non-stdlib dependencies. Uses a temp dir so it never touches
    # the real state/ directory.
    # ---------------------------------------------------------------- #
    import tempfile
    import shutil as _shutil

    print("\nSmoke-testing halt alerting...")
    tmp_dir = tempfile.mkdtemp(prefix="riskguardian_alert_test_")
    try:
        active_path = os.path.join(tmp_dir, "ACTIVE_HALT.json")
        history_path = os.path.join(tmp_dir, "halt_alerts.log.jsonl")

        def _read_history():
            if not os.path.exists(history_path):
                return []
            with open(history_path) as f:
                return [json.loads(line) for line in f if line.strip()]

        limits = RiskLimits(
            max_daily_loss_pct=0.05, max_drawdown_halt=0.15, min_hold_ticks=0,
            kill_switch_path="/tmp/nonexistent_kill_switch_alert_test",
            active_halt_path=active_path, alert_history_path=history_path,
            halt_reminder_interval_sec=60.0,
        )
        g = RiskGuardian(limits, starting_equity=10.0)

        # 1. Triggering a halt writes the active marker + one history entry.
        ok, reason = g.check(0.4, current_tick=1, equity=9.0, peak_equity=10.0, max_drawdown=0.0, book_is_stale=False)
        assert not ok and reason == "max_daily_loss_breached"
        assert os.path.exists(active_path), "active-halt marker was not created on trigger"
        active = RiskGuardian.read_active_halt(active_path)
        assert active is not None and active["reason_code"] == "max_daily_loss_breached"
        hist = _read_history()
        assert len(hist) == 1 and hist[0]["event"] == "triggered", hist
        print("halt trigger -> marker + 1 history entry: OK")

        # 2. THE ACTUAL BUG BEING FIXED: calling check() again while still
        # halted must reach _maybe_send_reminder from inside the early-return
        # branch, not skip it. Immediately re-checking (no time elapsed) must
        # NOT fire a second alert yet (reminder interval not reached).
        ok, reason = g.check(0.4, current_tick=2, equity=9.0, peak_equity=10.0, max_drawdown=0.0, book_is_stale=False)
        assert not ok and reason == "halted_previously:max_daily_loss_breached"
        hist = _read_history()
        assert len(hist) == 1, f"expected no new alert yet, got {len(hist)} entries"
        print("repeat check while halted, before interval -> no duplicate alert: OK")

        # 3. Jump time forward past the reminder interval and check again --
        # this must now produce a second ('reminder') history entry. If the
        # early-return branch didn't call _maybe_send_reminder, this fails.
        real_time = time.time
        try:
            time.time = lambda: real_time() + 3600  # +1 hour, well past 60s interval
            ok, reason = g.check(0.4, current_tick=3, equity=9.0, peak_equity=10.0, max_drawdown=0.0, book_is_stale=False)
        finally:
            time.time = real_time
        assert not ok and reason == "halted_previously:max_daily_loss_breached"
        hist = _read_history()
        assert len(hist) == 2 and hist[1]["event"] == "reminder", hist
        print("repeat check while halted, past interval -> reminder fired: OK")

        # 4. check_order_allowed()'s early-return must independently reach the
        # same reminder path -- it's a separate call site, not a shared one.
        real_time = time.time
        try:
            time.time = lambda: real_time() + 7200
            ok, reason = g.check_order_allowed("buy", 1.0, 100.0)
        finally:
            time.time = real_time
        assert not ok
        hist = _read_history()
        assert len(hist) == 3 and hist[2]["event"] == "reminder", hist
        print("check_order_allowed reminder path -> OK")

        # 5. reset_halt() must remove the active marker (not recreate it) and
        # append exactly one 'resolved' entry -- the bug this guards against
        # is _fire_alert unconditionally rewriting the active file regardless
        # of which event fired, which would leave a stale 'still halted'
        # marker sitting right after a reset.
        g.reset_halt()
        assert not os.path.exists(active_path), "active-halt marker survived reset_halt()"
        assert RiskGuardian.read_active_halt(active_path) is None
        hist = _read_history()
        assert len(hist) == 4 and hist[3]["event"] == "resolved", hist
        print("reset_halt -> marker removed, exactly one 'resolved' entry: OK")

        # 6. Post-reset, the guardian behaves as not-halted again -- checked
        # with equity actually back above the daily-loss threshold. (Checking
        # with the *same* still-breaching equity would correctly re-halt --
        # reset_halt() clears the halt flag, not the day's real P&L, and it
        # shouldn't: forgetting a real loss just because a human cleared the
        # halt would be a worse bug than the one this section is testing for.)
        ok, reason = g.check(0.4, current_tick=200, equity=10.0, peak_equity=10.0, max_drawdown=0.0, book_is_stale=False)
        assert ok, f"guardian still refusing after reset_halt: {reason}"
        print("post-reset check() approves again (equity recovered): OK")
    finally:
        _shutil.rmtree(tmp_dir, ignore_errors=True)

    print("\nAll RiskGuardian alerting assertions passed.")
