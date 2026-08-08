"""
Recklessness OOM Watchdog: Host-Level Systemd Daemon for Project TradeJack.
Phase 3 SOTA: Implements dmesg vs docker logs parsing to eliminate Host RAM false positives.
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
        """Locks the child into Tier 3 via the oom_penalties table."""
        db_path = os.path.join(self.state_dir, f"child_{child_id}", "ledger.sqlite")
        current_time = time.time()
        lock_until = current_time + self.lockout_duration_seconds
        
        logger.warning(f"OOM WATCHDOG FIRED on Child {child_id} ({container_name}): {reason}. Enforcing 24h Tier 3 Lock.")
        
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
            
        try:
            conn = sqlite3.connect(db_path, timeout=10)
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("CREATE TABLE IF NOT EXISTS oom_penalties (id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp REAL, reason TEXT, lock_until REAL)")
            conn.execute("""
                INSERT INTO oom_penalties (timestamp, reason, lock_until)
                VALUES (?, ?, ?)
            """, (current_time, reason, lock_until))
            
            # Phase 3 Event Sourcing Sync: Apply the $10 penalty via Tax Bridge so agent deducts natively
            conn.execute("CREATE TABLE IF NOT EXISTS tax_assessments (id INTEGER PRIMARY KEY AUTOINCREMENT, child_id INTEGER, timestamp REAL, amount REAL, reason TEXT)")
            conn.execute("""
                INSERT INTO tax_assessments (child_id, timestamp, amount, reason)
                VALUES (?, ?, ?, ?)
            """, (child_id, current_time, self.penalty_dollars, "OOM_RECKLESSNESS_PENALTY"))
            
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Error applying OOM penalty to Child {child_id}: {e}")

    def start_monitoring(self):
        if self._monitor_thread and self._monitor_thread.is_alive(): return
        self._stop_event.clear()
        self._monitor_thread = threading.Thread(target=self._docker_event_loop, daemon=True)
        self._monitor_thread.start()
        logger.info("OOM Watchdog monitoring thread started.")

    def stop_monitoring(self):
        self._stop_event.set()
        if self._monitor_thread:
            self._monitor_thread.join(timeout=2.0)

    def _docker_event_loop(self):
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
                    container_name = event.get("Actor", {}).get("Attributes", {}).get("name", "")
                    exit_code = event.get("Actor", {}).get("Attributes", {}).get("exitCode", "")
                    status = event.get("status", "")
                    
                    if container_name.startswith("swarm_child_"):
                        try:
                            child_id = int(container_name.split("_")[-1])
                        except ValueError:
                            continue
                            
                        if status == "oom" or exit_code == "137":
                            # SOTA Fix: Parse Logs for True CUDA vs Host RAM False Positive
                            try:
                                dlog = subprocess.run(["docker", "logs", "--tail", "50", container_name], capture_output=True, text=True).stderr
                                if "CUDA out of memory" in dlog or "CUDAOutOfMemoryError" in dlog:
                                    self.apply_oom_penalty(child_id, container_name, "PyTorch_CUDA_OOM")
                                    continue
                            except Exception:
                                pass
                                
                            try:
                                dmesg = subprocess.run("dmesg | tail -n 50", shell=True, capture_output=True, text=True).stdout
                                if "Out of memory: Killed process" in dmesg and "python" in dmesg:
                                    logger.critical(f"HOST RAM OOM DETECTED! Kernel killed {container_name}. Pausing spawning.")
                                    continue
                            except Exception:
                                pass
                except json.JSONDecodeError:
                    continue
                except Exception as e:
                    logger.error(f"Error parsing docker event: {e}")
        except Exception as e:
            logger.error(f"OOM Watchdog event loop crashed: {e}")
