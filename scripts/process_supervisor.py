"""
Process Supervisor for genesis_prime.py --mode {paper, testnet, live}.

Closes the gap named in the Phase 1 upgrade plan: nothing restarted the trading
process if it exited -- an unhandled exception, the feed giving up after repeated
failed resyncs, an OOM kill, anything -- there was zero code or docs covering this
for the active pipeline. A supervisor that restarts the process is only safe once a
graceful stop signal actually produces a graceful stop, which is why the SIGTERM
handling fix in genesis_prime.py's run_live() came first, in the same pass as this
file -- this supervisor sends SIGTERM (not SIGKILL) for its own graceful shutdown,
and that would have silently skipped server.stop()/trainer.stop() before that fix.

Design choices, and why:

  - Exponential backoff between restarts (capped), not immediate restart. A crash
    loop against a REAL exchange connection restarting every second is a much worse
    failure mode than a crash loop that isn't running at all -- rapid reconnect/
    reorder-placement attempts can trip exchange rate limits or worse, place
    duplicate orders during a half-completed reconciliation. Slowing down on
    repeated failure is a deliberate safety property, not just politeness.

  - A circuit breaker (max restarts within a rolling window) that STOPS trying and
    pages a human, rather than restarting forever. An infinite restart loop that
    quietly keeps failing is arguably worse than the process just staying down --
    "silently stopped, discover it three days later" (the failure mode this whole
    Phase 1 item exists to close) becomes "silently crash-looping against a live
    exchange for three days," which is not an improvement.

  - This script is a portable fallback, not a replacement for OS-level supervision.
    systemd (`Restart=always`) or Docker (`--restart unless-stopped`) should be
    preferred in production when available -- they run as PID 1 / a system service
    and survive this script's own process dying, which this script cannot do for
    itself. See docs/PROCESS_SUPERVISION.md for both paths and when to use each.

Usage:
    python -m scripts.process_supervisor --mode testnet --symbol BTC-USDT --capital 100.0
    python -m scripts.process_supervisor --mode live --symbol BTC-USDT --capital 100.0 \\
        --max-restarts 5 --restart-window-minutes 60

Stopping the supervisor (and, via the fix above, the child) gracefully:
    SIGINT (Ctrl+C) or SIGTERM to the supervisor process -- it forwards SIGTERM to
    the child and waits for it to exit before exiting itself, rather than just
    killing its own process and abandoning the child.
"""

import os
import sys
import json
import time
import signal
import logging
import argparse
import subprocess
from collections import deque
from datetime import datetime, timezone
from typing import List, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] (ProcessSupervisor) %(message)s")
logger = logging.getLogger("ProcessSupervisor")

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


class CircuitBreakerOpen(Exception):
    """Raised when too many restarts have happened within the rolling window."""


class ProcessSupervisor:
    def __init__(
        self,
        command: List[str],
        state_dir: str,
        initial_backoff_sec: float = 5.0,
        max_backoff_sec: float = 300.0,
        backoff_multiplier: float = 2.0,
        max_restarts: int = 5,
        restart_window_minutes: float = 60.0,
        graceful_shutdown_timeout_sec: float = 30.0,
    ):
        self.command = command
        self.state_dir = state_dir
        self.initial_backoff_sec = initial_backoff_sec
        self.max_backoff_sec = max_backoff_sec
        self.backoff_multiplier = backoff_multiplier
        self.max_restarts = max_restarts
        self.restart_window = restart_window_minutes * 60.0
        self.graceful_shutdown_timeout_sec = graceful_shutdown_timeout_sec

        self._restart_timestamps: deque = deque()  # monotonic times of recent restarts
        self._current_backoff = initial_backoff_sec
        self._proc: Optional[subprocess.Popen] = None
        self._shutting_down = False

        self._log_path = os.path.join(self.state_dir, "supervisor", "restart_log.jsonl")
        self._alert_path = os.path.join(self.state_dir, "supervisor", "CIRCUIT_BREAKER_OPEN.json")
        os.makedirs(os.path.dirname(self._log_path), exist_ok=True)

    # ── Restart bookkeeping ──────────────────────────────────────────────

    def _record_restart(self, exit_code: Optional[int], reason: str) -> None:
        now = time.monotonic()
        self._restart_timestamps.append(now)
        # Drop timestamps outside the rolling window.
        cutoff = now - self.restart_window
        while self._restart_timestamps and self._restart_timestamps[0] < cutoff:
            self._restart_timestamps.popleft()

        record = {
            "utc": datetime.now(timezone.utc).isoformat(),
            "exit_code": exit_code,
            "reason": reason,
            "restarts_in_window": len(self._restart_timestamps),
            "backoff_sec_next": self._current_backoff,
        }
        with open(self._log_path, "a") as f:
            f.write(json.dumps(record) + "\n")
        logger.warning(f"Restart recorded: {record}")

    def _circuit_breaker_tripped(self) -> bool:
        return len(self._restart_timestamps) >= self.max_restarts

    def _open_circuit_breaker(self) -> None:
        payload = {
            "opened_at_utc": datetime.now(timezone.utc).isoformat(),
            "restarts_in_window": len(self._restart_timestamps),
            "max_restarts": self.max_restarts,
            "restart_window_minutes": self.restart_window / 60.0,
            "message": (
                f"{len(self._restart_timestamps)} restarts within "
                f"{self.restart_window / 60.0:.0f} minute(s) -- exceeds max_restarts="
                f"{self.max_restarts}. Supervisor has STOPPED trying and will not "
                f"restart the process again on its own. This is deliberate: a crash "
                f"loop against a real exchange connection is worse than staying down. "
                f"A human must investigate the cause (see restart_log.jsonl in this "
                f"same directory) and delete this file to allow the supervisor to "
                f"resume."
            ),
        }
        with open(self._alert_path, "w") as f:
            json.dump(payload, f, indent=2)
        logger.critical(payload["message"])
        logger.critical(f"Alert written to {self._alert_path}")

    # ── Child process lifecycle ──────────────────────────────────────────

    def _spawn(self) -> subprocess.Popen:
        logger.info(f"Spawning: {' '.join(self.command)}")
        return subprocess.Popen(self.command, cwd=PROJECT_ROOT)

    def _poll_wait(self, proc: subprocess.Popen, timeout: float) -> Optional[int]:
        """Waits for `proc` to exit within `timeout`, via poll() in a sleep loop
        rather than Popen.wait(timeout=...). Deliberate: this supervisor calls into
        the same Popen object from both the main loop and from a SIGTERM/SIGINT
        signal handler, and Popen.wait() is NOT safe to call concurrently/reentrantly
        on the same object from a signal handler while another wait() on it is
        already blocked in the same thread -- confirmed by reproducing an actual
        hang under that exact pattern (main loop blocked in self._proc.wait() with no
        timeout, signal handler firing mid-wait and calling self._proc.wait(timeout=N)
        again) while building this. poll() (a single non-blocking os.waitpid(WNOHANG))
        has no such issue, so both the main loop and the signal handler exclusively
        use this helper instead of ever calling .wait() directly."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            code = proc.poll()
            if code is not None:
                return code
            time.sleep(0.1)
        return proc.poll()

    def _terminate_child_gracefully(self) -> None:
        if self._proc is None or self._proc.poll() is not None:
            return
        logger.info(f"Sending SIGTERM to child pid={self._proc.pid} and waiting up to "
                    f"{self.graceful_shutdown_timeout_sec}s for graceful shutdown.")
        self._proc.send_signal(signal.SIGTERM)
        code = self._poll_wait(self._proc, self.graceful_shutdown_timeout_sec)
        if code is not None:
            logger.info(f"Child exited gracefully with code {code}.")
        else:
            logger.warning(
                f"Child did not exit within {self.graceful_shutdown_timeout_sec}s of SIGTERM "
                f"-- sending SIGKILL. This means the graceful-shutdown path in genesis_prime.py "
                f"did not complete in time; investigate why before assuming state is clean."
            )
            self._proc.kill()
            self._poll_wait(self._proc, 5.0)

    def _install_signal_handlers(self) -> None:
        def _handle(signum, frame):
            sig_name = signal.Signals(signum).name
            logger.info(f"Supervisor received {sig_name} -- shutting down supervised child, then exiting.")
            self._shutting_down = True
            self._terminate_child_gracefully()

        signal.signal(signal.SIGINT, _handle)
        signal.signal(signal.SIGTERM, _handle)

    # ── Main loop ─────────────────────────────────────────────────────────

    def run(self) -> int:
        """Runs until the supervisor is asked to shut down, the circuit breaker trips,
        or the child exits with code 0 (treated as an intentional, non-crash stop --
        e.g. an operator-triggered clean shutdown of the child itself, not via this
        supervisor's own signal handler). Returns a process exit code."""
        if os.path.exists(self._alert_path):
            logger.critical(
                f"Circuit breaker alert file already present at {self._alert_path} -- "
                f"refusing to start. Investigate, then delete that file to proceed."
            )
            return 1

        self._install_signal_handlers()

        while not self._shutting_down:
            self._proc = self._spawn()
            # Poll rather than a blocking wait() with no timeout -- see
            # _poll_wait()'s docstring for why: this loop and the SIGTERM/SIGINT
            # signal handler must never both be inside Popen.wait() on the same
            # object at once.
            while True:
                code = self._proc.poll()
                if code is not None:
                    exit_code = code
                    break
                if self._shutting_down:
                    exit_code = None
                    break
                time.sleep(0.2)

            if self._shutting_down:
                # Our own signal handler already requested/performed shutdown of the
                # child (or the child happened to exit right as we were shutting
                # down) -- this is a clean stop, not a crash to restart from.
                return 0

            if exit_code == 0:
                logger.info("Child exited with code 0 (clean, intentional stop) -- supervisor exiting, not restarting.")
                return 0

            self._record_restart(exit_code, reason=f"child exited with code {exit_code}")

            if self._circuit_breaker_tripped():
                self._open_circuit_breaker()
                return 1

            logger.warning(f"Child exited with code {exit_code}. Restarting in {self._current_backoff:.0f}s "
                            f"({len(self._restart_timestamps)}/{self.max_restarts} restarts used in the current "
                            f"{self.restart_window / 60.0:.0f}-minute window).")
            time.sleep(self._current_backoff)
            self._current_backoff = min(self._current_backoff * self.backoff_multiplier, self.max_backoff_sec)

        return 0


def build_command(args: argparse.Namespace) -> List[str]:
    cmd = [
        sys.executable, "-m", "scripts.genesis_prime",
        "--mode", args.mode,
        "--symbol", args.symbol,
        "--capital", str(args.capital),
    ]
    return cmd


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Restart-with-backoff supervisor for genesis_prime.py.")
    parser.add_argument("--mode", choices=["paper", "testnet", "live"], required=True)
    parser.add_argument("--symbol", type=str, default="BTC-USDT")
    parser.add_argument("--capital", type=float, default=100.0)
    parser.add_argument("--state-dir", type=str, default=os.path.join(PROJECT_ROOT, "state"))
    parser.add_argument("--initial-backoff-sec", type=float, default=5.0)
    parser.add_argument("--max-backoff-sec", type=float, default=300.0)
    parser.add_argument("--backoff-multiplier", type=float, default=2.0)
    parser.add_argument("--max-restarts", type=int, default=5,
                         help="Circuit breaker: max restarts allowed within --restart-window-minutes.")
    parser.add_argument("--restart-window-minutes", type=float, default=60.0)
    parser.add_argument("--graceful-shutdown-timeout-sec", type=float, default=30.0)
    args = parser.parse_args()

    supervisor = ProcessSupervisor(
        command=build_command(args),
        state_dir=args.state_dir,
        initial_backoff_sec=args.initial_backoff_sec,
        max_backoff_sec=args.max_backoff_sec,
        backoff_multiplier=args.backoff_multiplier,
        max_restarts=args.max_restarts,
        restart_window_minutes=args.restart_window_minutes,
        graceful_shutdown_timeout_sec=args.graceful_shutdown_timeout_sec,
    )
    sys.exit(supervisor.run())
