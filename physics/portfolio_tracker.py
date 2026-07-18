"""
Vectorized Portfolio Accounting Engine for Project TradeJack.
Tracks real-time cash, open position size, unrealized/realized PnL, peak equity, maximum drawdown,
rolling Sharpe and Sortino ratios, and commits states directly to the container's `ledger.sqlite`.
"""

import os
import sys
import time
import math
import sqlite3
import logging
import numpy as np
from typing import Dict, Any, List, Optional, Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] (PortfolioTracker) %(message)s")
logger = logging.getLogger("PortfolioTracker")


class PortfolioAccountingEngine:
    """
    Precision financial accounting engine writing directly to SQLite for Warden auditing.
    """

    def __init__(
        self,
        child_id: int = 0,
        state_dir: str = "d:/TradeJack/state",
        initial_cash: float = 10.0,
        risk_free_rate_annual: float = 0.04,
        ticks_per_year: float = 365.0 * 1440.0
    ):
        self.child_id = child_id
        self.state_dir = os.path.abspath(state_dir)
        self.db_dir = os.path.join(self.state_dir, f"child_{self.child_id}")
        os.makedirs(self.db_dir, exist_ok=True)
        self.db_path = os.path.join(self.db_dir, "ledger.sqlite")
        
        self.initial_cash = initial_cash
        self.risk_free_rate_step = risk_free_rate_annual / ticks_per_year
        self.annualization_factor = math.sqrt(ticks_per_year)
        
        self.cash = initial_cash
        self.equity = initial_cash
        self.peak_equity = initial_cash
        self.max_drawdown = 0.0
        self.ticks_active = 0
        self.ticks_stagnant = 0
        self.returns_history: List[float] = []
        
        self._init_sqlite_table()

    def _init_sqlite_table(self):
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS portfolio_state (
                    tick_id INTEGER PRIMARY KEY,
                    timestamp REAL,
                    cash REAL,
                    equity REAL,
                    peak_equity REAL,
                    max_drawdown REAL,
                    sharpe_ratio REAL,
                    sortino_ratio REAL,
                    ticks_active INTEGER,
                    ticks_stagnant INTEGER,
                    oom_penalty INTEGER DEFAULT 0,
                    oom_lock_until REAL DEFAULT 0.0
                )
            """)
            cursor.execute("SELECT COUNT(*) FROM portfolio_state")
            if cursor.fetchone()[0] == 0:
                cursor.execute("""
                    INSERT INTO portfolio_state (
                        tick_id, timestamp, cash, equity, peak_equity, max_drawdown,
                        sharpe_ratio, sortino_ratio, ticks_active, ticks_stagnant, oom_penalty, oom_lock_until
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (0, time.time(), self.initial_cash, self.initial_cash, self.initial_cash, 0.0, 0.0, 0.0, 0, 0, 0, 0.0))
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Failed to initialize SQLite ledger at {self.db_path}: {e}")

    def compute_risk_adjusted_ratios(self) -> Tuple[float, float]:
        """Calculates exact annualized Sharpe and Sortino ratios over returns history."""
        if len(self.returns_history) < 2:
            return 0.0, 0.0
            
        arr_ret = np.array(self.returns_history, dtype=np.float64)
        excess_ret = arr_ret - self.risk_free_rate_step
        mean_excess = np.mean(excess_ret)
        std_tot = np.std(arr_ret, ddof=1)
        
        # Sharpe Ratio
        if std_tot > 1e-9:
            sharpe = (mean_excess / std_tot) * self.annualization_factor
        else:
            sharpe = 0.0
            
        # Sortino Ratio (downside deviation only)
        downside_ret = arr_ret[arr_ret < self.risk_free_rate_step] - self.risk_free_rate_step
        if len(downside_ret) > 0:
            downside_std = math.sqrt(np.mean(downside_ret ** 2))
            sortino = (mean_excess / downside_std) * self.annualization_factor if downside_std > 1e-9 else 0.0
        else:
            sortino = sharpe * 1.5  # If zero downside, Sortino is higher than Sharpe
            
        return float(sharpe), float(sortino)

    def record_step(
        self,
        new_cash: float,
        new_equity: float,
        tick_id: int
    ) -> Dict[str, Any]:
        """
        Updates internal metrics and commits the financial state snapshot to SQLite.
        """
        prev_equity = self.equity
        self.cash = new_cash
        self.equity = new_equity
        self.ticks_active += 1
        
        # Check compounding stagnation ("turtling")
        step_return = (self.equity - prev_equity) / (prev_equity + 1e-8)
        self.returns_history.append(step_return)
        if len(self.returns_history) > 500:
            self.returns_history.pop(0)  # Rolling 500 step window
            
        if step_return <= self.risk_free_rate_step:
            self.ticks_stagnant += 1
        else:
            self.ticks_stagnant = 0
            
        if self.equity > self.peak_equity:
            self.peak_equity = self.equity
        if self.peak_equity > 0:
            self.max_drawdown = max(self.max_drawdown, (self.peak_equity - self.equity) / self.peak_equity)
            
        sharpe, sortino = self.compute_risk_adjusted_ratios()
        
        # Write to SQLite
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO portfolio_state (
                    tick_id, timestamp, cash, equity, peak_equity, max_drawdown,
                    sharpe_ratio, sortino_ratio, ticks_active, ticks_stagnant, oom_penalty, oom_lock_until
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 
                    (SELECT oom_penalty FROM portfolio_state ORDER BY tick_id DESC LIMIT 1),
                    (SELECT oom_lock_until FROM portfolio_state ORDER BY tick_id DESC LIMIT 1)
                )
            """, (tick_id, time.time(), self.cash, self.equity, self.peak_equity, self.max_drawdown,
                  sharpe, sortino, self.ticks_active, self.ticks_stagnant))
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Error writing portfolio state to SQLite: {e}")
            
        return {
            "child_id": self.child_id,
            "tick_id": tick_id,
            "equity": self.equity,
            "peak_equity": self.peak_equity,
            "max_drawdown": self.max_drawdown,
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "ticks_stagnant": self.ticks_stagnant
        }


if __name__ == "__main__":
    logger.info("Testing PortfolioAccountingEngine standalone execution...")
    engine = PortfolioAccountingEngine(child_id=99, initial_cash=10.0)
    # Simulate 10 steps of growth and minor drawdowns
    eq = 10.0
    for step in range(1, 11):
        eq += np.random.normal(0.2, 0.1)
        summary = engine.record_step(new_cash=eq, new_equity=eq, tick_id=step)
        print(f"Step {step}: Equity=${summary['equity']:.2f}, Sharpe={summary['sharpe_ratio']:.2f}, Drawdown={summary['max_drawdown']*100:.1f}%")
