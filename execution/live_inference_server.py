"""
Live-paper inference server: wires binance_live_feed -> streaming_features ->
a frozen (never-trained-during-this-loop) model -> risk_guardian ->
paper_exchange into one continuous loop, and logs every decision — approved,
rejected, and filled — for review afterward via session_report.py.

This is the Phase-4-per-the-plan-discussion piece: real Binance market data,
synthetic wallet, no third-party demo broker. It answers "if these frozen
weights were released right now, what would happen" as honestly as an
in-process simulation can, which is why every friction (latency, fees, visible
depth, staleness) is modeled rather than assumed away.

Nothing here trains the model or places a real order. Model weights are loaded
once at startup and never updated (mirrors scripts/deploy_config.py's frozen-
deployment philosophy) — this is deliberately a *validation* stage, not the
self-RL training loop (that's swarm/rl_trainer.py + training/continuous_trainer.py,
separate work).
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import time
import json
from datetime import datetime, timedelta, timezone
import sqlite3
import logging
import argparse
import asyncio
import threading
from collections import deque
from typing import Any, Deque, Dict, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] (LiveInference) %(message)s")
logger = logging.getLogger("LiveInference")

import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from stable_baselines3 import PPO, SAC, DQN
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False

from swarm.model_registry import REGISTRY
from swarm.baselines import MomentumBaseline
from scripts.deploy_config import DEPLOY_CONFIG
from execution.binance_live_feed import BinanceLiveDepthFeed, SyntheticReplayFeed
from execution.streaming_features import StreamingFeatureEngine
from execution.paper_exchange import PaperExchange
from execution.risk_guardian import RiskGuardian, RiskLimits


def _check_promotion_gate(weights_path: str) -> None:
    """
    PHASE 0 FIX: nothing previously checked whether the checkpoint sitting at
    `weights_path` (i.e. DEPLOY_CONFIG.frozen_model_path, what every mode --
    paper, testnet, live -- loads) had actually passed its own validation
    gate. Verified directly against this repo's own state: the checkpoint at
    state/deployed/weights_promoted.zip was written by `train_and_promote.py
    --force` with promotion_log.jsonl recording `airgap_passed: false,
    avg_sharpe: -0.073` after only 256 training timesteps (one line in
    state/training_logs/progress.csv) -- a rejected smoke-test checkpoint,
    force-promoted anyway, silently loaded as "the live model" by every mode.
    `--force` is a legitimate, documented human override for iterating on the
    pipeline -- the bug is that nothing downstream re-surfaces that this
    override was used once the checkpoint is actually about to trade. This
    makes that loud instead of silent: refuses to load a checkpoint whose own
    most recent promotion record says it failed the gate, unless the
    operator explicitly acknowledges it via TRADEJACK_ALLOW_FAILED_PROMOTION=1
    (deliberately not a code-level flag -- an env var so it can't be
    accidentally left on in a config file that gets reused).
    """
    log_path = os.path.join(os.path.dirname(os.path.abspath(weights_path)), "promotion_log.jsonl")
    if not os.path.exists(log_path):
        return  # no promotion history alongside this checkpoint -- nothing to gate on

    def _norm(p: str) -> str:
        # Promotion logs can be written on Windows (backslash separators) and
        # read back on Linux (or vice versa) -- os.path.normpath alone does
        # NOT convert '\' to '/' on a non-Windows host, so a naive normpath
        # comparison silently never matches a Windows-written log entry.
        # Caught by this fix's own test against this repo's real log file.
        return os.path.normpath(p.replace("\\", "/"))

    target_norm = _norm(weights_path)
    last_matching = None
    try:
        with open(log_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                rec_target = rec.get("target_path", "")
                if _norm(rec_target) == target_norm:
                    last_matching = rec  # jsonl is append-only chronological; last match wins
    except Exception as e:
        logger.warning(f"Could not read promotion log at {log_path}: {e}")
        return

    if last_matching is not None and last_matching.get("airgap_passed") is False:
        if os.environ.get("TRADEJACK_ALLOW_FAILED_PROMOTION") == "1":
            logger.critical(
                f"LOADING A CHECKPOINT THAT FAILED VALIDATION: {weights_path} "
                f"(avg_sharpe={last_matching.get('avg_sharpe')}, "
                f"max_drawdown={last_matching.get('max_drawdown')}) -- proceeding "
                f"ONLY because TRADEJACK_ALLOW_FAILED_PROMOTION=1 is set. This model "
                f"lost the statistical validation the pipeline exists to enforce."
            )
            return
        raise RuntimeError(
            f"Refusing to load '{weights_path}': its own promotion record shows "
            f"airgap_passed=False (avg_sharpe={last_matching.get('avg_sharpe')}, "
            f"max_drawdown={last_matching.get('max_drawdown')}). This checkpoint was "
            f"force-promoted despite failing validation and should not be trusted for "
            f"paper, testnet, or live trading. Retrain and let it pass the gate "
            f"normally, or set TRADEJACK_ALLOW_FAILED_PROMOTION=1 if you specifically "
            f"intend to run this known-bad checkpoint anyway (e.g. to test the "
            f"pipeline itself)."
        )


def load_frozen_model(weights_path: str, model_name: str = "PPO-DilatedCNN", input_dim: int = 5) -> Any:
    """
    Loads frozen weights for live paper inference.
    Supports:
      1. SB3 checkpoints (.zip) loaded via PPO/SAC/DQN
      2. PyTorch state_dict (.pt) checkpoints
      3. Rule-based baselines (Momentum-Baseline, BuyAndHold-Baseline)
      4. Safe fallback if promoted checkpoint is not yet generated
    """
    if weights_path:
        _check_promotion_gate(weights_path)

    card = REGISTRY.get_model_card(model_name)
    algo_class = card.algo_class if card else "PPO"

    # 1. Attempt SB3 checkpoint load
    algo_map = {"PPO": PPO, "SAC": SAC, "DQN": DQN} if SB3_AVAILABLE else {}
    target_cls = algo_map.get(algo_class)

    candidates = [weights_path]
    if weights_path and not weights_path.endswith(".zip"):
        candidates.append(weights_path + ".zip")

    for path in candidates:
        if path and os.path.exists(path):
            if SB3_AVAILABLE and target_cls is not None:
                try:
                    model = target_cls.load(path, device="cpu")
                    logger.info(f"Loaded frozen SB3 model from {path} ({model_name})")
                    return model
                except Exception as e:
                    logger.warning(f"Failed to load SB3 checkpoint {path}: {e}")

            if TORCH_AVAILABLE:
                try:
                    state = torch.load(path, map_location="cpu")
                    model = REGISTRY.build_model(model_name, input_dim=input_dim)
                    if hasattr(model, "net"):
                        sd = state.get("model_state_dict", state) if isinstance(state, dict) else state
                        model.net.load_state_dict(sd, strict=False)
                        model.net.eval()
                        logger.info(f"Loaded frozen PyTorch weights from {path}")
                        return model
                except Exception as e:
                    logger.warning(f"Failed to load PyTorch checkpoint {path}: {e}")

    # 2. Check if baseline or rule-based model
    if card and card.algo_class in ("rule", "static"):
        logger.info(f"Building baseline model '{model_name}' directly from registry.")
        return REGISTRY.build_model(model_name)

    # 3. Fallback when weights_path does not exist
    logger.warning(
        f"No promoted weights found at '{weights_path}'. Running fallback MomentumBaseline "
        f"for pipeline verification. Run scripts/train_and_promote.py to train and deploy real weights."
    )
    return MomentumBaseline()


def _testnet_gate_path(state_dir: str) -> str:
    return os.path.join(state_dir, "testnet_first_session.json")


def _enforce_testnet_gate(exchange_mode: str, state_dir: str) -> None:
    """
    PHASE 1 FIX: `DeploymentConfig.min_weeks_testnet_before_live` used to be asserted
    at construction (must be >= 1) and referenced only in comments -- nothing measured
    elapsed testnet time or blocked `exchange_mode="live"` if it hadn't been met. It
    was a documentation convention for the human operator, not a software gate, despite
    the field's name implying otherwise. This makes it real:

      - The first time this server runs with exchange_mode="testnet", it records that
        moment (UTC) to `{state_dir}/testnet_first_session.json`. Repeat testnet runs
        do not reset this -- the clock starts on first testnet use, not last.
      - Before constructing a REAL exchange_mode="live" connection, it checks that at
        least `DEPLOY_CONFIG.min_weeks_testnet_before_live` weeks have elapsed since
        that first recorded testnet session. No recorded testnet history at all is
        treated the same as "not enough" -- refused, not silently allowed.
      - `TRADEJACK_SKIP_TESTNET_GATE=1` overrides it, for the same reason
        TRADEJACK_ALLOW_FAILED_PROMOTION exists: an operator can always choose to
        proceed anyway, but the default is refuse-and-explain, not silently permit.

    Deliberately per-state_dir, not global: different accounts/symbols using different
    state_dirs each need their own testnet track record before going live independently.
    """
    gate_path = _testnet_gate_path(state_dir)

    if exchange_mode == "testnet":
        if not os.path.exists(gate_path):
            os.makedirs(state_dir, exist_ok=True)
            payload = {"first_testnet_utc": datetime.now(timezone.utc).isoformat()}
            with open(gate_path, "w") as f:
                json.dump(payload, f)
            logger.info(f"Recorded first testnet session at {payload['first_testnet_utc']} -> {gate_path}")
        return

    if exchange_mode != "live":
        return  # paper mode: nothing to gate

    min_weeks = getattr(DEPLOY_CONFIG, "min_weeks_testnet_before_live", 2)
    required = timedelta(weeks=min_weeks)

    if os.environ.get("TRADEJACK_SKIP_TESTNET_GATE") == "1":
        logger.critical(
            "TESTNET GATE BYPASSED via TRADEJACK_SKIP_TESTNET_GATE=1 -- proceeding to "
            "exchange_mode='live' regardless of recorded testnet history. This was the "
            "one thing on the upgrade plan explicitly stated as 'cannot be shortcut with "
            "more code' -- bypassing it is an operator decision, not a software one."
        )
        return

    if not os.path.exists(gate_path):
        raise RuntimeError(
            f"Refusing to start exchange_mode='live': no testnet session has ever been "
            f"recorded at '{gate_path}'. Run with exchange_mode='testnet' first for at "
            f"least {min_weeks} week(s) before going live, or set "
            f"TRADEJACK_SKIP_TESTNET_GATE=1 if you specifically intend to skip this."
        )

    with open(gate_path, "r") as f:
        recorded = json.load(f)
    first_testnet = datetime.fromisoformat(recorded["first_testnet_utc"])
    elapsed = datetime.now(timezone.utc) - first_testnet
    if elapsed < required:
        remaining = required - elapsed
        raise RuntimeError(
            f"Refusing to start exchange_mode='live': only {elapsed.days} day(s) of "
            f"testnet history recorded (first session {recorded['first_testnet_utc']}), "
            f"{min_weeks} week(s) required -- {remaining.days} day(s) remaining. Set "
            f"TRADEJACK_SKIP_TESTNET_GATE=1 if you specifically intend to skip this."
        )
    logger.info(f"Testnet gate satisfied: {elapsed.days} day(s) of history since {recorded['first_testnet_utc']}.")


class LivePaperInferenceServer:
    def __init__(
        self,
        symbol: str = "BTC-USDT",
        weights_path: Optional[str] = None,
        model_name: Optional[str] = None,
        initial_cash: float = 10.0,
        seq_len: int = 64,
        use_synthetic_feed: bool = False,
        state_dir: str = "state",
        account_id: int = 900,
        owns_feed: bool = True,
        decision_interval_ticks: int = 1,
        risk_limits_override: Optional[Dict[str, Any]] = None,
        exchange_mode: Optional[str] = None,
        broker: Optional[str] = None,
        use_composition_layer: bool = False,
        predictor: Optional[Any] = None,
        toxicity_symbol: Optional[str] = None,
    ):
        """
        owns_feed: True (default) means this server opens its own live
            BinanceLiveDepthFeed connection, as before. Set False when a
            MultiSleeveOrchestrator is fanning out ONE shared connection to
            several sleeves — every sleeve on the same symbol opening its own
            socket wastes a connection and, more importantly, means each
            sleeve's book state can drift out of sync with the others' by a
            message or two, which is exactly the kind of subtle inconsistency
            an aggregate risk check across sleeves should not have to reason
            about. When False, the orchestrator is responsible for calling
            `await server._on_update(kind, payload)` directly for every
            message, and `run()` must not be called on this instance.

        decision_interval_ticks: how many depth updates must arrive before
            `_maybe_act()` recomputes a decision, once the observation buffer
            is full. 1 = react on every update (a scalping/HFT sleeve, bounded
            by Binance's 100ms depth stream — see MultiSleeveOrchestrator's
            docstring for why that is NOT the same thing as colocated HFT).
            Larger values (e.g. 300 = roughly every 30s at 100ms/update, 3000
            = roughly every 5min) give a position/swing sleeve a slower,
            calmer decision cadence without needing a second feed or a
            different feature pipeline — it sees every update (so its
            features stay current) but only acts periodically.

        use_composition_layer: False (default) preserves the exact prior
            behavior — the RL model's raw action, clipped to [-1, 1], goes
            straight to the risk check and the exchange, unchanged. True
            routes the RL action through execution.live_composer's
            LiveOrderComposer first: an optional predictor's confirmation/
            disagreement, a risk-budget throttle, a toxicity throttle, and a
            loss-streak throttle all get to scale (never redirect) the size
            before it's submitted. See _maybe_act()'s composition block for
            exactly how the two systems' different units (fraction-of-equity
            vs. asset-quantity) are reconciled, and docs/COMPOSITION_LAYER.md
            for the design.
        predictor: an optional fitted `data_forge.predictor.SignalPredictor`.
            Only consulted when use_composition_layer=True. None is a fully
            valid, safe choice — the composer's "unconfirmed" bucket handles
            "no predictor" the same way it handles "predictor says flat" or
            "predictor call failed": a flat multiplier, not a crash and not a
            veto.
        toxicity_symbol: only consulted when use_composition_layer=True — see
            LiveOrderComposer's own docstring for its auto-load/caching
            behavior. None (default) disables the toxicity throttle entirely.
        """
        self.symbol = symbol
        self.seq_len = seq_len
        self.tick = 0
        self.updates_since_last_decision = 0
        self.decision_interval_ticks = max(1, decision_interval_ticks)
        self.obs_buffer: Deque[np.ndarray] = deque(maxlen=seq_len)

        self.model_name = model_name or DEPLOY_CONFIG.model_name
        self.is_running = False
        self.model = load_frozen_model(
            weights_path or DEPLOY_CONFIG.frozen_model_path,
            self.model_name,
        )
        self.feature_engine = StreamingFeatureEngine()

        # THE FIX: this used to unconditionally construct PaperExchange here,
        # regardless of DEPLOY_CONFIG.exchange_mode -- meaning "live" mode
        # never actually placed a real order under any configuration, despite
        # execution/exchange_adapter.py's BinanceSpotAdapter being fully built
        # and unit-tested in isolation. Confirmed by grepping every
        # BinanceSpotAdapter( construction site in the repo before this fix:
        # only the dashboard's read-only balance display and the adapter's
        # own self-tests ever instantiated it. See
        # execution/live_exchange_bridge.py's module docstring for the full
        # story and why a bridge class, not a rewrite of _maybe_act(), was
        # the right fix.
        self.exchange_mode = exchange_mode or DEPLOY_CONFIG.exchange_mode
        if self.exchange_mode == "paper":
            self.exchange = PaperExchange(
                symbol=symbol, initial_cash=initial_cash, state_dir=state_dir, account_id=account_id
            )
        else:
            _enforce_testnet_gate(self.exchange_mode, state_dir)
            broker = (broker or getattr(DEPLOY_CONFIG, "broker", "binance")).lower()
            logger.warning(
                f"exchange_mode='{self.exchange_mode}', broker='{broker}' -- constructing a REAL "
                f"exchange connection (testnet={self.exchange_mode != 'live'}). Orders placed by this "
                f"server will be REAL if exchange_mode == 'live'. LiveExchangeBridge has been tested "
                f"against a mock adapter only (see its module docstring) -- verify on testnet "
                f"extensively before this."
            )
            from execution.live_exchange_bridge import LiveExchangeBridge
            if broker == "oanda":
                # FX pairs (EURUSD, GBPUSD, ...) go through OANDA -- see
                # execution/oanda_adapter.py's module docstring for why OANDA
                # rather than MetaTrader (the official MT5 package is
                # Windows-only; OANDA's v20 API is native REST/streaming, no
                # terminal required, matching this project's headless-Linux
                # asyncio deployment model). Symbol stays in this project's
                # no-separator convention (EURUSD); OandaAdapter itself
                # converts to OANDA's EUR_USD internally.
                from execution.oanda_adapter import OandaAdapter
                adapter = OandaAdapter(symbol=symbol, environment=("live" if self.exchange_mode == "live" else "practice"))
                bridge_symbol = symbol
            elif broker == "binance":
                from execution.exchange_adapter import BinanceSpotAdapter
                adapter = BinanceSpotAdapter(testnet=(self.exchange_mode != "live"))
                bridge_symbol = symbol.replace("-", "/") if "-" in symbol and "/" not in symbol else symbol
            else:
                raise ValueError(f"Unknown broker '{broker}' -- expected 'binance' or 'oanda'.")
            self.exchange = LiveExchangeBridge(
                symbol=bridge_symbol, adapter=adapter, state_dir=state_dir, account_id=account_id,
            )

        limits_kwargs = dict(
            max_position_fraction=DEPLOY_CONFIG.max_position_fraction,
            min_hold_ticks=DEPLOY_CONFIG.min_hold_ticks,
            max_daily_loss_pct=DEPLOY_CONFIG.max_daily_loss_pct,
            max_drawdown_halt=DEPLOY_CONFIG.max_drawdown_halt,
            kill_switch_path=os.path.join(state_dir, "KILL_SWITCH"),
            # Anchored to state_dir for the same reason kill_switch_path is:
            # RiskLimits' own defaults for these two are bare relative paths
            # ("state/ACTIVE_HALT.json", "state/halt_alerts.log.jsonl"), which
            # resolve against the process's CURRENT WORKING DIRECTORY at
            # launch time, not against state_dir -- a real, previously-missed
            # bug, since kill_switch_path got this treatment but these two
            # didn't. If this process is ever launched from a different
            # working directory than state_dir's parent (a realistic scenario
            # for a systemd-managed service -- see docs/PROCESS_SUPERVISION.md),
            # the halt marker and alert history would silently land somewhere
            # other than where the dashboard's /api/risk-halt looks for them,
            # defeating the entire point of the sticky-halt-plus-alerting
            # design: a halt would still happen correctly (the in-memory
            # RiskGuardian state is unaffected either way), but nothing
            # watching the configured state_dir would see it happen.
            active_halt_path=os.path.join(state_dir, "ACTIVE_HALT.json"),
            alert_history_path=os.path.join(state_dir, "halt_alerts.log.jsonl"),
        )
        limits_kwargs.update(risk_limits_override or {})
        # Pass the real starting equity explicitly where it's actually known
        # at construction time (paper mode: PaperExchange.initial_cash is
        # exact). In live/testnet mode LiveExchangeBridge.initial_cash is
        # still 0.0 here (unset until its first background reconciliation
        # completes), so leave it None -- RiskGuardian's own lazy-capture
        # logic (_reset_daily_if_needed) picks up the true baseline from the
        # first real `equity` value passed into check() instead of trusting
        # a number that hasn't been fetched from the exchange yet.
        known_starting_equity = self.exchange.initial_cash if self.exchange_mode == "paper" else None
        self.risk = RiskGuardian(RiskLimits(**limits_kwargs), starting_equity=known_starting_equity)

        self.use_composition_layer = use_composition_layer
        self.predictor = predictor
        self.composer: Optional[Any] = None
        if use_composition_layer:
            from execution.live_composer import LiveOrderComposer

            self.composer = LiveOrderComposer(risk_guardian=self.risk, toxicity_symbol=toxicity_symbol)

        self.owns_feed = owns_feed
        self.use_synthetic_feed = use_synthetic_feed
        if not owns_feed:
            self.feed = None  # orchestrator drives this instance's _on_update directly
        elif use_synthetic_feed:
            logger.warning("Running against SyntheticReplayFeed — NOT real market data. Logic-testing mode only.")
            self.feed = SyntheticReplayFeed(symbol=symbol)
        else:
            self.feed = BinanceLiveDepthFeed(symbol=symbol, on_update=self._on_update)

        self._init_decision_ledger(state_dir, account_id)

        # Guards self.model during a hot swap. See hot_swap_model() below —
        # this is the method training/continuous_trainer.py's
        # _promote_champion() looks for via hasattr(inference_server,
        # "hot_swap_model"). It didn't exist at all before this fix, so every
        # promotion silently no-opped on the live side: the hasattr() guard
        # meant it never crashed, it just never actually updated the running
        # model either.
        self._model_lock = threading.Lock()
        self.model_version = 0
        self.last_promotion_time: Optional[float] = None

    @classmethod
    def from_config(cls, cfg: Any):
        """Construct LivePaperInferenceServer from DeploymentConfig."""
        # PHASE 0 FIX: this used to derive use_synthetic_feed from
        # exchange_mode == "paper" -- meaning genesis_prime.py's documented
        # `--mode paper` entrypoint (paper being the default and the
        # recommended way to evaluate a model before risking anything) ran
        # against SyntheticReplayFeed's random walk, never real prices, by
        # construction. Any Sharpe/drawdown/win-rate evidence gathered that
        # way was evaluating the model against noise, not the market. These
        # are genuinely independent choices -- exchange_mode picks where
        # ORDERS go, use_synthetic_feed should pick where PRICES come from --
        # and are now read as two separate config fields accordingly.
        server = cls(
            symbol=getattr(cfg, "symbol", "BTC-USDT"),
            weights_path=getattr(cfg, "frozen_model_path", None),
            model_name=getattr(cfg, "model_name", "PPO-DilatedCNN"),
            initial_cash=getattr(cfg, "starting_capital", 100.0),
            use_synthetic_feed=getattr(cfg, "use_synthetic_feed", False),
            broker=getattr(cfg, "broker", None),
        )
        return server

    def get_status(self) -> Dict[str, Any]:
        """Return server status."""
        return {
            "model": self.model_name,
            "model_version": self.model_version,
            "last_promotion_time": self.last_promotion_time,
            "running": self.is_running,
            "symbol": self.symbol,
            "tick": self.tick,
            "equity": self.exchange.accounting.equity if hasattr(self.exchange, "accounting") else self.exchange.initial_cash,
        }

    def hot_swap_model(self, new_weights_path: str, new_model_name: Optional[str] = None) -> bool:
        """
        Atomically swaps the frozen inference model for a newly-promoted
        checkpoint. Loads the new model into a local variable FIRST and only
        assigns it to self.model if loading succeeds and it's a real trained
        model — not load_frozen_model()'s MomentumBaseline fallback, which
        exists for a good reason at server startup (better to trade a known
        baseline than crash) but must NOT be allowed to silently substitute
        for a promotion whose checkpoint path was wrong or whose copy step
        failed. That should be a loud failure, not a quiet strategy downgrade.
        Returns True on a successful swap, False otherwise — never raises, so
        a failed promotion can't halt trading that was already working.
        """
        model_name = new_model_name or self.model_name

        candidate_paths = [new_weights_path]
        if new_weights_path and not new_weights_path.endswith(".zip"):
            candidate_paths.append(new_weights_path + ".zip")
        if not any(os.path.exists(p) for p in candidate_paths):
            logger.error(
                f"Hot-swap FAILED: no checkpoint found at '{new_weights_path}' (or .zip). "
                f"Keeping current model_version={self.model_version}."
            )
            return False

        try:
            candidate = load_frozen_model(new_weights_path, model_name)
        except Exception as e:
            logger.error(f"Hot-swap FAILED to load '{new_weights_path}': {e}. Keeping current model.")
            return False

        if candidate is None or isinstance(candidate, MomentumBaseline):
            logger.error(f"Hot-swap FAILED: loader fell back to a baseline for '{new_weights_path}'.")
            return False

        with self._model_lock:
            self.model = candidate
            self.model_name = model_name
            self.model_version += 1
            self.last_promotion_time = time.time()

        logger.warning(
            f"HOT-SWAP COMPLETE: now running model_version={self.model_version} "
            f"({model_name}) loaded from {new_weights_path}"
        )
        return True

    def _init_decision_ledger(self, state_dir: str, account_id: int):
        db_dir = os.path.join(os.path.abspath(state_dir), f"child_{account_id}")
        os.makedirs(db_dir, exist_ok=True)
        self.decision_db_path = os.path.join(db_dir, "decisions.sqlite")
        conn = sqlite3.connect(self.decision_db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS decisions (
                id INTEGER PRIMARY KEY AUTOINCREMENT, tick INTEGER, timestamp REAL,
                target_frac REAL, approved INTEGER, reject_reason TEXT, equity REAL,
                mid_price REAL
            )
        """)
        try:
            conn.execute("ALTER TABLE decisions ADD COLUMN mid_price REAL")
        except Exception:
            pass
        conn.commit()
        conn.close()

    def _log_decision(self, target_frac: float, approved: bool, reason: Optional[str], mid_price: float = 0.0):
        try:
            conn = sqlite3.connect(self.decision_db_path, timeout=5)
            conn.execute(
                "INSERT INTO decisions (tick, timestamp, target_frac, approved, reject_reason, equity, mid_price) VALUES (?,?,?,?,?,?,?)",
                (self.tick, time.time(), target_frac, int(approved), reason, self.exchange.accounting.equity, mid_price),
            )
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Decision ledger write failed: {e}")

    def _think(self) -> float:
        with self._model_lock:
            model = self.model  # local reference: safe even if hot_swap_model() reassigns self.model mid-call

        seq = np.stack(list(self.obs_buffer), axis=0)  # (seq_len, 5)
        mean = seq.mean(axis=0)
        std = seq.std(axis=0) + 1e-8
        norm = ((seq - mean) / std).astype(np.float32)

        # Standard Gymnasium observation dict expected by SB3 & Baselines
        port_state = np.array([
            self.exchange.accounting.cash / max(self.exchange.initial_cash, 1.0),
            self.exchange.position_qty,
            self.exchange.accounting.equity / max(self.exchange.initial_cash, 1.0),
            self.exchange.accounting.max_drawdown,
        ], dtype=np.float32)

        obs = {
            "lob_sequence": norm,
            "portfolio_state": port_state,
        }

        # 1. SB3 or Baseline predict(obs) interface
        if hasattr(model, "predict"):
            try:
                action, _ = model.predict(obs, deterministic=True)
                if isinstance(action, (list, np.ndarray)):
                    return float(np.clip(action[0], -1.0, 1.0))
                return float(np.clip(action, -1.0, 1.0))
            except Exception as e:
                logger.debug(f"predict(obs) failed ({e}), checking legacy paths")

        # 2. PyTorch neural network forward pass
        if TORCH_AVAILABLE and hasattr(model, "net"):
            t_in = torch.from_numpy(norm).unsqueeze(0)
            with torch.no_grad():
                out = model.forward(t_in)
            out_arr = out.numpy() if hasattr(out, "numpy") else np.array(out)
            if out_arr.shape[-1] == 3:
                idx = int(np.argmax(out_arr, axis=-1).reshape(-1)[0])
                action_val = {0: 0.0, 1: 1.0, 2: -1.0}.get(idx, 0.0)
            else:
                action_val = float(np.mean(out_arr))
            return float(np.clip(action_val, -1.0, 1.0))

        # 3. Callable model.forward(norm)
        if hasattr(model, "forward"):
            try:
                out = model.forward(norm)
                out_arr = np.array(out)
                if out_arr.shape[-1] == 3:
                    idx = int(np.argmax(out_arr, axis=-1).reshape(-1)[0])
                    action_val = {0: 0.0, 1: 1.0, 2: -1.0}.get(idx, 0.0)
                else:
                    action_val = float(np.mean(out_arr))
                return float(np.clip(action_val, -1.0, 1.0))
            except Exception:
                pass

        return 0.0

    async def _maybe_act(self):
        if len(self.obs_buffer) < self.seq_len:
            return

        self.updates_since_last_decision += 1
        if self.updates_since_last_decision < self.decision_interval_ticks:
            return  # features stay current every update; decisions happen on this sleeve's own cadence
        self.updates_since_last_decision = 0

        self.tick += 1
        target_frac = self._think()

        approved, reason = self.risk.check(
            target_frac=target_frac,
            current_tick=self.tick,
            equity=self.exchange.accounting.equity,
            peak_equity=self.exchange.accounting.peak_equity,
            max_drawdown=self.exchange.accounting.max_drawdown,
            book_is_stale=self.exchange.book_is_stale(self.risk.limits.max_book_staleness_sec),
        )
        mid = getattr(self.exchange, "last_mid_price", 0.0)
        self._log_decision(target_frac, approved, reason, mid_price=mid)

        if not approved:
            if reason not in ("min_hold_ticks_not_elapsed",):  # this one fires constantly by design, don't spam
                logger.info(f"Tick {self.tick}: decision REJECTED ({reason}), target_frac={target_frac:.3f}")
            return

        final_frac = target_frac
        if self.composer is not None:
            # Reconciles two different unit spaces: _think() and submit_target_position()
            # work in fraction-of-equity space ([-1, 1]); LiveOrderComposer/SignalComposer
            # work in asset-quantity space (base_qty, so its internal risk_check_fn can
            # validate the actual proposed order cost). The conversion below derives
            # base_qty as "the quantity RL's raw target_frac would imply at current
            # equity/price", then converts the composed result back the same way --
            # this is the exact fraction<->quantity round-trip, not an approximation:
            #   base_qty = |target_frac| * equity / price
            #   proposed_qty = base_qty * intent.size_fraction   (done inside compose())
            #   => final_frac = intent.direction * |target_frac| * intent.size_fraction
            # is algebraically final_frac = proposed_qty * price / equity, i.e. the
            # actual composed order re-expressed as a fraction of equity, sign included.
            equity = self.exchange.accounting.equity
            if equity <= 0 or mid <= 0:
                logger.warning(f"Tick {self.tick}: equity/price non-positive (equity={equity}, price={mid}) "
                                f"-- skipping composition this tick, no order.")
                return

            base_qty = abs(target_frac) * equity / mid
            # Raw (pre-z-score) feature history -- self.obs_buffer already stores
            # StreamingFeatureEngine's physics-schema rows (close_price/volume/ofi/
            # vpin_50/kyles_lambda-shaped) unmodified; the z-score normalization in
            # _think() is a temporary copy made only for the RL model's input, so this
            # is the same feature space feature_engineering.py's physics table uses,
            # not the RL-specific normalized view.
            feature_window = np.stack(list(self.obs_buffer), axis=0)

            intent = self.composer.compose_order(
                rl_action=target_frac,
                base_qty=base_qty,
                price=mid,
                predictor=self.predictor,
                feature_window=feature_window,
            )

            if intent.bucket != "flat_rl" and intent.direction == 0:
                # Composition suppressed a nonzero RL intent entirely (predictor_locked,
                # risk_vetoed, or a throttle stack that scaled size to ~0) -- this means
                # "don't place the trade RL wanted", not "close whatever is currently
                # open". Explicitly skip submission rather than calling
                # submit_target_position(0.0), which would force-flatten an existing
                # position based on a decision that was never about the existing
                # position in the first place.
                logger.info(f"Tick {self.tick}: composition suppressed order [{intent.bucket}] {intent.reason}")
                return

            final_frac = float(np.clip(intent.direction * abs(target_frac) * intent.size_fraction, -1.0, 1.0))
            if intent.bucket not in ("flat_rl",):
                logger.info(
                    f"Tick {self.tick}: composed [{intent.bucket}] raw_target={target_frac:.3f} -> "
                    f"final={final_frac:.3f} (risk_budget={intent.risk_budget_scalar:.2f}, "
                    f"toxicity={intent.toxicity_scalar:.2f}, performance={intent.performance_scalar:.2f}) "
                    f"{intent.reason}"
                )

        self.risk.record_order_submitted(self.tick)
        result = await self.exchange.submit_target_position(final_frac)
        if result.filled_qty != 0.0 or result.rejected_reason:
            logger.info(
                f"Tick {self.tick}: target={final_frac:.3f} filled={result.filled_qty:.6f} "
                f"@ {result.avg_price:.2f} fee={result.fee_paid:.4f} latency={result.latency_sec*1000:.0f}ms "
                f"reason={result.rejected_reason} equity=${self.exchange.accounting.equity:.4f}"
            )

    async def _on_update(self, kind: str, payload: Dict[str, Any]):
        if kind == "depth":
            self.exchange.on_depth_update(payload)
            feat = self.feature_engine.on_depth(payload)
            if feat is not None:
                self.obs_buffer.append(feat)
                await self._maybe_act()
        elif kind == "trade":
            self.feature_engine.on_trade(payload)

    async def run(self, duration_sec: Optional[float] = None):
        if not self.owns_feed:
            raise RuntimeError(
                "This server was constructed with owns_feed=False — it's meant to be driven by "
                "a MultiSleeveOrchestrator calling _on_update() directly, not by calling run() itself."
            )
        logger.info(
            f"Starting live-paper session: symbol={self.symbol} "
            f"cash=${self.exchange.initial_cash:.2f}{'  (real exchange balance pending reconciliation)' if self.exchange_mode != 'paper' else ''} "
            f"model={DEPLOY_CONFIG.model_name} "
            f"synthetic_feed={self.use_synthetic_feed} exchange_mode={self.exchange_mode}"
        )
        try:
            if self.use_synthetic_feed:
                await self.feed.run(self._on_update, duration_sec=duration_sec)
            else:
                await self.feed.run(duration_sec=duration_sec)
        finally:
            self.exchange.close()
            logger.info(
                f"Session ended. Final equity: ${self.exchange.accounting.equity:.4f} "
                f"(started at ${self.exchange.initial_cash:.2f}). "
                f"Trades: {self.exchange.trade_count}. Total fees: ${self.exchange.total_fees_paid:.4f}. "
                f"Ledger: {self.exchange.fill_db_path} / {self.decision_db_path}"
            )

    async def run_forever(self):
        """
        BUG FOUND BY ACTUALLY RUNNING `genesis_prime.py --mode paper` (and
        testnet/live), not by reading either file: that orchestration code
        calls `server.run_forever()` and `server.stop()`, but this class only
        ever had `run(duration_sec=...)` — no `run_forever` or `stop` existed
        at all. Every one of genesis_prime.py's paper/testnet/live modes
        crashed immediately with AttributeError on the very first live
        session, meaning that code path had never actually been run
        end-to-end before. This is a thin, honest wrapper — no new behavior,
        just the interface genesis_prime.py already assumed existed.
        """
        await self.run(duration_sec=None)

    def stop(self):
        """Companion to run_forever() — signals the underlying feed to stop,
        which unblocks run()/run_forever()'s await on the next message."""
        if self.feed is not None:
            self.feed.stop()
        else:
            logger.warning("stop() called but this server has no owned feed (owns_feed=False?) — nothing to stop here.")


def _parse_args():
    p = argparse.ArgumentParser(description="TradeJack live-paper inference session")
    p.add_argument("--symbol", default="BTC-USDT")
    p.add_argument("--cash", type=float, default=10.0)
    p.add_argument("--duration-sec", type=float, default=None)
    p.add_argument("--weights-path", default=None)
    p.add_argument("--model-name", default=None)
    p.add_argument("--synthetic", action="store_true", help="use SyntheticReplayFeed instead of real Binance data (logic testing only)")
    p.add_argument("--state-dir", default="state")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    server = LivePaperInferenceServer(
        symbol=args.symbol,
        weights_path=args.weights_path,
        model_name=args.model_name,
        initial_cash=args.cash,
        use_synthetic_feed=args.synthetic,
        state_dir=args.state_dir,
    )
    asyncio.run(server.run(duration_sec=args.duration_sec))


# Alias for backward compatibility with scripts/run_all_checks.py
LiveInferenceServer = LivePaperInferenceServer
