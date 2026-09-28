"""
TrainingProgressLedger: durable, queryable storage for RL training progress
across cycles and agents.

Why this needed to exist: swarm/rl_trainer.py's TrainingMetricsCallback keeps
a `metrics_history` list, but it's in-memory only, tied to one OnlineRLTrainer
object's lifetime. For a dashboard to show "how has this agent's training
progressed" durably -- surviving process restarts, viewable while the
continuous training loop is still running in a separate process -- training
progress needs to land somewhere queryable from outside that process. This is
the same pattern as execution/paper_exchange.py's fills ledger and
execution/live_inference_server.py's decisions ledger: SQLite, one writer
(the training loop), any number of readers (a dashboard, a report script).
"""

import os
import time
import sqlite3
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger("TrainingProgressLedger")


class TrainingProgressLedger:
    def __init__(self, state_dir: str = "state"):
        self.db_path = os.path.join(os.path.abspath(state_dir), "training_progress.sqlite")
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS cycles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL, agent_id TEXT, label TEXT, model_name TEXT, algo_class TEXT,
                cumulative_timesteps INTEGER, sortino_ratio REAL, best_sortino REAL, equity REAL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL, agent_id TEXT, event_type TEXT, detail TEXT
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_cycles_agent ON cycles(agent_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_events_agent ON events(agent_id)")
        conn.commit()
        conn.close()

    def record_cycle(
        self,
        agent_id: Any,
        label: str,
        model_name: str,
        algo_class: str,
        cumulative_timesteps: int,
        sortino_ratio: float,
        best_sortino: float,
        equity: float,
    ):
        try:
            conn = sqlite3.connect(self.db_path, timeout=5)
            conn.execute(
                """INSERT INTO cycles
                (timestamp, agent_id, label, model_name, algo_class, cumulative_timesteps, sortino_ratio, best_sortino, equity)
                VALUES (?,?,?,?,?,?,?,?,?)""",
                (time.time(), str(agent_id), label, model_name, algo_class, cumulative_timesteps, sortino_ratio, best_sortino, equity),
            )
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Failed to record training cycle: {e}")

    def record_event(self, agent_id: Any, event_type: str, detail: str = ""):
        """event_type: 'promotion' | 'plasticity_reset' | 'pbt_exploit' | etc.
        Recorded as its own table (not a column on cycles) since events don't
        happen once per cycle -- a promotion affects the whole tournament,
        not one agent's row."""
        try:
            conn = sqlite3.connect(self.db_path, timeout=5)
            conn.execute(
                "INSERT INTO events (timestamp, agent_id, event_type, detail) VALUES (?,?,?,?)",
                (time.time(), str(agent_id), event_type, detail),
            )
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Failed to record training event: {e}")

    def get_cycles(self, agent_id: Optional[Any] = None, limit: int = 500) -> List[Dict[str, Any]]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        if agent_id is not None:
            rows = conn.execute(
                "SELECT * FROM cycles WHERE agent_id=? ORDER BY id DESC LIMIT ?", (str(agent_id), limit)
            ).fetchall()
        else:
            rows = conn.execute("SELECT * FROM cycles ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        conn.close()
        return [dict(r) for r in reversed(rows)]

    def get_events(self, agent_id: Optional[Any] = None, limit: int = 200) -> List[Dict[str, Any]]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        if agent_id is not None:
            rows = conn.execute(
                "SELECT * FROM events WHERE agent_id=? ORDER BY id DESC LIMIT ?", (str(agent_id), limit)
            ).fetchall()
        else:
            rows = conn.execute("SELECT * FROM events ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        conn.close()
        return [dict(r) for r in reversed(rows)]


if __name__ == "__main__":
    import shutil
    test_dir = "/tmp/tj_progress_ledger_test"
    shutil.rmtree(test_dir, ignore_errors=True)
    ledger = TrainingProgressLedger(state_dir=f"{test_dir}/state")
    for i in range(5):
        ledger.record_cycle("0", "PPO-CNN", "PPO-DilatedCNN", "PPO", i * 1000, 0.1 * i, 0.1 * i, 10.0 + i * 0.1)
    ledger.record_event("0", "promotion", "cycle 3")
    ledger.record_event("0", "plasticity_reset", "reset 2/6 head layers")
    cycles = ledger.get_cycles("0")
    events = ledger.get_events("0")
    print(f"Recorded {len(cycles)} cycles, {len(events)} events")
    assert len(cycles) == 5
    assert len(events) == 2
    assert cycles[-1]["sortino_ratio"] == 0.4
    print("TrainingProgressLedger smoke test passed.")
    shutil.rmtree(test_dir, ignore_errors=True)
