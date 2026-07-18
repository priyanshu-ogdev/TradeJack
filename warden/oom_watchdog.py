"""
Recklessness OOM Watchdog: Host-Level Systemd Daemon for Project TradeJack.
Monitors Docker event streams (`oom` / `die` exit code 137) and CUDA SIGKILLs across Child containers.
If a Child agent recklessly exhausts its assigned VRAM slice during `self-mod/` adaptation, the Watchdog:
1. Docks its SQLite ledger with a -$10.00 Recklessness Penalty.
2. Hard-locks its memory tier to Tier 3 (Inference-Only / 0GB Train VRAM) for 24 simulated hours.
"""

import os
import sys
import time
import json
import sqlite3
import logging
import subprocess
import threading
from typing import Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] (OOMWatchdog) %(message)s")
logger = logging.getLogger("OOMWatchdog")


class RecklessnessWatchdog:
    """
    Monitors container events and penalizes reckless memory usage.
    """

    def __init__(
        self,
        state_dir: str = "d:/TradeJack/state",
        penalty_dollars: float = 10.0,
        lockout_duration_seconds: float = 86400.0
    ):
        self.state_dir = os.path.abspath(state_dir)
        self.penalty_dollars = penalty_dollars
        self.lockout_duration_seconds = lockout_duration_seconds
        self.log_dir = os.path.join(self.state_dir, "logs")
        os.makedirs(self.log_dir, exist_ok=True)
        self.recklessness_log_path = os.path.join(self.log_dir, "recklessness.log")
        self._stop_event = threading.Event()
        self._monitor_thread: Optional[threading.Thread] = None

    def apply_oom_penalty(self, child_id: int, container_name: str, reason: str = "CUDA_OOM_KILLED"):
        """
        Docks $10.00 from the child's portfolio state table and sets `oom_penalty=1` and `oom_lock_until` timer.
        """
        db_path = os.path.join(self.state_dir, f"child_{child_id}", "ledger.sqlite")
        current_time = time.time()
        lock_until = current_time + self.lockout_duration_seconds
        
        logger.warning(f"OOM WATCHDOG FIRED on Child {child_id} ({container_name}): {reason}. Applying -${self.penalty_dollars:.2f} penalty and 24h Tier 3 lock.")
        
        # Log incident
        try:
            with open(self.recklessness_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "timestamp": current_time,
                    "child_id": child_id,
                    "container_name": container_name,
                    "reason": reason,
                    "penalty_dollars": self.penalty_dollars,
                    "lock_until": lock_until
                }) + "\n")
        except Exception as e:
            logger.error(f"Failed to write to recklessness log: {e}")
            
        if not os.path.exists(db_path):
            logger.error(f"Ledger file not found for Child {child_id}: {db_path}")
            return
            
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            # Deduct penalty and apply lock on latest row
            cursor.execute("""
                UPDATE portfolio_state
                SET cash = cash - ?,
                    equity = equity - ?,
                    oom_penalty = 1,
                    oom_lock_until = ?
                WHERE tick_id = (SELECT MAX(tick_id) FROM portfolio_state)
            """, (self.penalty_dollars, self.penalty_dollars, lock_until))
            
            # Check resulting equity
            cursor.execute("SELECT equity FROM portfolio_state WHERE tick_id = (SELECT MAX(tick_id) FROM portfolio_state)")
            row = cursor.fetchone()
            conn.commit()
            conn.close()
            
            if row and row[0] <= 0.0:
                logger.warning(f"Child {child_id} became insolvent after Recklessness Penalty (Equity: ${row[0]:.2f}).")
        except Exception as e:
            logger.error(f"Error applying OOM penalty to Child {child_id} ledger: {e}")

    def start_monitoring(self):
        """Starts background Docker event monitoring loop."""
        if self._monitor_thread and self._monitor_thread.is_alive():
            return
        self._stop_event.clear()
        self._monitor_thread = threading.Thread(target=self._docker_event_loop, daemon=True)
        self._monitor_thread.start()
        logger.info("OOM Watchdog monitoring thread started.")

    def stop_monitoring(self):
        """Stops background monitoring loop."""
        self._stop_event.set()
        if self._monitor_thread:
            self._monitor_thread.join(timeout=2.0)
            logger.info("OOM Watchdog stopped.")

    def _docker_event_loop(self):
        """
        Subprocesses `docker events --filter 'event=oom' --filter 'event=die' --format '{{json .}}'`
        and checks for `swarm_child_*` containers terminating with exit code 137 or OOM flags.
        """
        cmd = ["docker", "events", "--filter", "event=oom", "--filter", "event=die", "--format", "{{json .}}"]
        try:
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
            while not self._stop_event.is_set():
                line = process.stdout.readline() if process.stdout else ""
                if not line:
                    time.sleep(0.5)
                    continue
                try:
                    event = json.loads(line.strip())
                    actor = event.get("Actor", {})
                    attributes = actor.get("Attributes", {})
                    container_name = attributes.get("name", "")
                    exit_code = attributes.get("exitCode", "")
                    status = event.get("status", "")
                    
                    if container_name.startswith("swarm_child_"):
                        # Extract child ID
                        parts = container_name.split("_")
                        try:
                            child_id = int(parts[-1])
                        except ValueError:
                            continue
                            
                        if status == "oom" or exit_code == "137":
                            self.apply_oom_penalty(child_id, container_name, reason=f"Docker {status} (exitCode {exit_code})")
                except json.JSONDecodeError:
                    continue
        except Exception as e:
            logger.debug(f"Docker event monitor loop exited: {e}")


if __name__ == "__main__":
    logger.info("Testing Recklessness Watchdog Standalone Penalty Injection...")
    watchdog = RecklessnessWatchdog(state_dir="d:/TradeJack/state")
    # Simulate child 0 getting OOM
    os.makedirs("d:/TradeJack/state/child_0", exist_ok=True)
    conn = sqlite3.connect("d:/TradeJack/state/child_0/ledger.sqlite")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS portfolio_state (
            tick_id INTEGER PRIMARY KEY, timestamp REAL, cash REAL, equity REAL,
            peak_equity REAL, max_drawdown REAL, sharpe_ratio REAL, sortino_ratio REAL,
            ticks_active INTEGER, ticks_stagnant INTEGER, oom_penalty INTEGER DEFAULT 0, oom_lock_until REAL DEFAULT 0.0
        )
    """)
    conn.execute("INSERT OR REPLACE INTO portfolio_state VALUES (1, ?, 100.0, 100.0, 100.0, 0.0, 1.5, 1.5, 10, 0, 0, 0.0)", (time.time(),))
    conn.commit()
    conn.close()
    
    watchdog.apply_oom_penalty(0, "swarm_child_0", "Simulated OOM Test")
