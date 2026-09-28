import os
import time
import math
import sqlite3
import logging
import threading
import queue
import numpy as np
from collections import deque
from typing import Dict, Any, Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] (PortfolioTracker) %(message)s")
logger = logging.getLogger("PortfolioTracker")

class PortfolioAccountingEngine:
    """
    Asynchronous, non-blocking financial accounting engine.
    Batches writes to SQLite via a background thread to prevent GPU I/O starvation.
    """
    def __init__(
        self,
        child_id: int = 0,
        state_dir: str = "state",
        initial_cash: float = 10.0,
        risk_free_rate_annual: float = 0.04,
        ticks_per_year: float = 365.0 * 1440.0,
        flush_interval_sec: float = 2.0
    ):
        self.child_id = child_id
        self.db_dir = os.path.join(os.path.abspath(state_dir), f"child_{self.child_id}")
        os.makedirs(self.db_dir, exist_ok=True)
        self.db_path = os.path.join(self.db_dir, "ledger.sqlite")
        
        self.initial_cash = initial_cash
        self.risk_free_rate_step = risk_free_rate_annual / ticks_per_year
        self.annualization_factor = math.sqrt(ticks_per_year)
        
        # State variables
        self.cash = initial_cash
        self.equity = initial_cash
        self.peak_equity = initial_cash
        self.max_drawdown = 0.0
        self.ticks_active = 0
        
        # Time-Dilation Stagnation Tracker
        self.last_hwm_market_timestamp = 0.0
        # REVIEW FIX (starvation-logic audit): tracks the market_timestamp of this
        # child's very first record_step() call. Needed to fix apply_survival_tax()'s
        # base tax -- see that method's docstring in warden/warden_core.py for the
        # full explanation of why ticks_active alone cannot be converted to elapsed
        # hours for this data.
        self.first_market_timestamp = 0.0
        
        # Event-Sourcing Tax Bridge
        self.last_processed_tax_id = 0
        self.pending_tax = 0.0
        
        # O(1) Rolling Window for Sortino Penalty
        self.returns_rolling = deque(maxlen=500)
        
        # O(1) Lifetime Tracking via Welford's Algorithm
        self.lifetime_n = 0
        self.lifetime_mean = 0.0
        self.lifetime_m2 = 0.0
        
        # Async Queue and Thread
        self.write_queue = queue.Queue()
        self.flush_interval = flush_interval_sec
        self._stop_event = threading.Event()
        self._worker = threading.Thread(target=self._flush_worker, daemon=True)
        self._worker.start()
        
        self._init_sqlite()

    def _init_sqlite(self):
        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute("PRAGMA journal_mode=WAL;")
            # SOTA Phase 3 schema
            conn.execute("""
                CREATE TABLE IF NOT EXISTS portfolio_state (
                    tick_id INTEGER PRIMARY KEY, timestamp REAL, cash REAL, equity REAL, 
                    peak_equity REAL, max_drawdown REAL, lifetime_sharpe REAL, 
                    rolling_sortino REAL, ticks_active INTEGER, last_hwm_market_timestamp REAL,
                    market_timestamp REAL, first_market_timestamp REAL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS tax_assessments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    child_id INTEGER,
                    market_timestamp TEXT,
                    tax_type TEXT,
                    amount REAL,
                    UNIQUE(child_id, market_timestamp, tax_type)
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS oom_penalties (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp REAL,
                    reason TEXT, lock_until REAL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS tier_transitions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp REAL,
                    tick_id INTEGER, from_tier INTEGER, to_tier INTEGER,
                    equity REAL, reason TEXT
                )
            """)
            # REVIEW FIX (starvation-logic audit): CREATE TABLE IF NOT EXISTS is a
            # no-op against an already-existing ledger from before this column was
            # added -- an ALTER TABLE migration is required for any child that was
            # spawned before this fix. Wrapped in try/except: SQLite raises if the
            # column already exists (a fresh ledger created with the new schema
            # above already has it), which is the expected, harmless case here.
            try:
                conn.execute("ALTER TABLE portfolio_state ADD COLUMN first_market_timestamp REAL DEFAULT 0.0")
            except sqlite3.OperationalError:
                pass
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"SQLite Init failed: {e}")

    def _welford_update(self, ret: float):
        self.lifetime_n += 1
        delta = ret - self.lifetime_mean
        self.lifetime_mean += delta / self.lifetime_n
        delta2 = ret - self.lifetime_mean
        self.lifetime_m2 += delta * delta2

    def _compute_ratios(self) -> Tuple[float, float]:
        if self.lifetime_n < 2:
            lt_sharpe = 0.0
        else:
            lt_var = self.lifetime_m2 / (self.lifetime_n - 1)
            lt_std = math.sqrt(lt_var) if lt_var > 0 else 1e-9
            lt_sharpe = ((self.lifetime_mean - self.risk_free_rate_step) / lt_std) * self.annualization_factor
            
        if len(self.returns_rolling) < 10:
            r_sortino = 0.0
        else:
            arr = np.array(self.returns_rolling)
            excess = np.mean(arr) - self.risk_free_rate_step
            downside = arr[arr < self.risk_free_rate_step] - self.risk_free_rate_step
            if len(downside) > 0:
                down_std = math.sqrt(np.mean(downside ** 2))
                r_sortino = (excess / down_std) * self.annualization_factor if down_std > 1e-9 else 0.0
            else:
                r_sortino = lt_sharpe * 1.5
                
        return float(lt_sharpe), float(r_sortino)

    def record_step(self, new_cash: float, new_equity: float, tick_id: int, market_timestamp: float) -> Dict[str, Any]:
        """O(1) State update. Deducts async taxes natively from memory."""
        # 1. Tax Reconciliation Bridge (Natively absorb Warden penalties)
        if self.pending_tax > 0.0:
            tax = self.pending_tax
            self.pending_tax = 0.0
            new_cash -= tax
            new_equity -= tax

        prev_equity = self.equity
        self.cash = new_cash
        self.equity = new_equity
        self.ticks_active += 1
        
        # 2. Time-Dilation Stagnation (Market Time High Water Mark)
        if self.ticks_active == 1:
            self.first_market_timestamp = market_timestamp
        if self.ticks_active == 1 or self.equity > self.peak_equity:
            self.peak_equity = self.equity
            self.last_hwm_market_timestamp = market_timestamp
            
        step_ret = (self.equity - prev_equity) / (prev_equity + 1e-8)
        self.returns_rolling.append(step_ret)
        self._welford_update(step_ret)
            
        if self.peak_equity > 0:
            self.max_drawdown = max(self.max_drawdown, (self.peak_equity - self.equity) / self.peak_equity)
            
        lt_sharpe, r_sortino = self._compute_ratios()
        
        self.write_queue.put((
            tick_id, time.time(), self.cash, self.equity, self.peak_equity, 
            self.max_drawdown, lt_sharpe, r_sortino, self.ticks_active, 
            self.last_hwm_market_timestamp, market_timestamp, self.first_market_timestamp
        ))
        
        return {
            "equity": self.equity,
            "cash": self.cash,
            "peak_equity": self.peak_equity,
            "max_drawdown": self.max_drawdown,
            "sharpe_ratio": lt_sharpe,        # alias for test_physics.py compatibility
            "lifetime_sharpe": lt_sharpe,
            "rolling_sortino": r_sortino,
            "ticks_active": self.ticks_active,
            "last_hwm": self.last_hwm_market_timestamp,
        }

    def _flush_worker(self):
        while not self._stop_event.is_set():
            batch = []
            try:
                batch.append(self.write_queue.get(timeout=self.flush_interval))
                while not self.write_queue.empty() and len(batch) < 1000:
                    batch.append(self.write_queue.get_nowait())
            except queue.Empty:
                continue
                
            if batch:
                try:
                    conn = sqlite3.connect(self.db_path, timeout=10)
                    conn.execute("PRAGMA journal_mode=WAL;")
                    conn.executemany("""
                        INSERT OR REPLACE INTO portfolio_state 
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, batch)
                    
                    # Read-Only Tax Sourcing
                    cursor = conn.cursor()
                    cursor.execute("SELECT MAX(id), SUM(amount) FROM tax_assessments WHERE id > ?", (self.last_processed_tax_id,))
                    row = cursor.fetchone()
                    if row and row[0] is not None:
                        self.last_processed_tax_id = row[0]
                        self.pending_tax += float(row[1])
                        
                    conn.commit()
                    # Prevent NVMe Checkpoint Bloat (SOTA fix)
                    conn.execute("PRAGMA wal_checkpoint(TRUNCATE);")
                    conn.close()
                except Exception as e:
                    logger.error(f"Async SQLite flush failed: {e}")
                    
            for _ in batch:
                self.write_queue.task_done()

    def close(self):
        self._stop_event.set()
        if self._worker.is_alive():
            self._worker.join(timeout=2.0)
        if not self.write_queue.empty():
            self._flush_worker()
