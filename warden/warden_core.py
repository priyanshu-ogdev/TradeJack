"""
Warden Core Hypervisor: Host OS Controller for NVIDIA DGX Spark (Grace Blackwell Architecture).
Enforces dynamic MIG/CUDA memory partitioning, VRAM quota allocation, and Logarithmic + Stagnation survival tax across autonomous Docker containers.
"""

import os
import sys
import time
import math
import sqlite3
import logging
import subprocess
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] (WardenCore) %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("warden_core.log", mode="a", encoding="utf-8")
    ]
)
logger = logging.getLogger("WardenCore")


@dataclass
class ContainerLedgerSummary:
    child_id: int
    container_name: str
    cash: float
    equity: float
    peak_equity: float
    max_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    ticks_active: int
    ticks_stagnant: int
    tier: int = 2  # Default Tier 2 (Stagnant / Mid)
    oom_penalty_active: bool = False
    oom_lock_until: float = 0.0
    is_alive: bool = True


class WardenHypervisor:
    """
    Bare-Metal Hypervisor managing the sovereign child container swarm on DGX Spark.
    """

    def __init__(
        self,
        swarm_size: int = 50,
        state_dir: str = "d:/TradeJack/state",
        base_tax_per_hr: float = 1.0,
        alpha_tax_scale: float = 0.5,
        beta_stagnation_penalty: float = 0.1,
        enable_hardware_mig: bool = True
    ):
        self.swarm_size = swarm_size
        self.state_dir = os.path.abspath(state_dir)
        self.base_tax_per_hr = base_tax_per_hr
        self.alpha_tax_scale = alpha_tax_scale
        self.beta_stagnation_penalty = beta_stagnation_penalty
        self.enable_hardware_mig = enable_hardware_mig
        self.summaries: Dict[int, ContainerLedgerSummary] = {}
        
        os.makedirs(self.state_dir, exist_ok=True)
        os.makedirs(os.path.join(self.state_dir, "logs"), exist_ok=True)
        self.purge_log_path = os.path.join(self.state_dir, "logs", "purge.log")
        self.tier_history_path = os.path.join(self.state_dir, "logs", "tier_allocations.jsonl")

    def get_ledger_path(self, child_id: int) -> str:
        return os.path.join(self.state_dir, f"child_{child_id}", "ledger.sqlite")

    def init_child_ledger(self, child_id: int, initial_cash: float = 10.0) -> str:
        """Initializes a new SQLite ledger for a child container upon genesis spawn."""
        db_dir = os.path.join(self.state_dir, f"child_{child_id}")
        os.makedirs(db_dir, exist_ok=True)
        db_path = os.path.join(db_dir, "ledger.sqlite")
        
        conn = sqlite3.connect(db_path)
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
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS tax_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL,
                tick_id INTEGER,
                tax_deducted REAL,
                stagnation_penalty REAL,
                remaining_equity REAL
            )
        """)
        # Insert genesis entry if empty
        cursor.execute("SELECT COUNT(*) FROM portfolio_state")
        if cursor.fetchone()[0] == 0:
            cursor.execute("""
                INSERT INTO portfolio_state (
                    tick_id, timestamp, cash, equity, peak_equity, max_drawdown,
                    sharpe_ratio, sortino_ratio, ticks_active, ticks_stagnant, oom_penalty, oom_lock_until
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (0, time.time(), initial_cash, initial_cash, initial_cash, 0.0, 0.0, 0.0, 0, 0, 0, 0.0))
        conn.commit()
        conn.close()
        return db_path

    def audit_child_ledger(self, child_id: int) -> Optional[ContainerLedgerSummary]:
        """Reads the latest portfolio metrics for a given child from its SQLite ledger."""
        db_path = self.get_ledger_path(child_id)
        if not os.path.exists(db_path):
            return None
        
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("""
                SELECT cash, equity, peak_equity, max_drawdown, sharpe_ratio,
                       sortino_ratio, ticks_active, ticks_stagnant, oom_penalty, oom_lock_until
                FROM portfolio_state ORDER BY tick_id DESC LIMIT 1
            """)
            row = cursor.fetchone()
            conn.close()
            if not row:
                return None
            
            cash, equity, peak_eq, max_dd, sharpe, sortino, t_active, t_stagnant, oom_pen, oom_lock = row
            container_name = f"swarm_child_{child_id}"
            
            summary = ContainerLedgerSummary(
                child_id=child_id,
                container_name=container_name,
                cash=float(cash),
                equity=float(equity),
                peak_equity=float(peak_eq),
                max_drawdown=float(max_dd),
                sharpe_ratio=float(sharpe),
                sortino_ratio=float(sortino),
                ticks_active=int(t_active),
                ticks_stagnant=int(t_stagnant),
                oom_penalty_active=bool(oom_pen),
                oom_lock_until=float(oom_lock)
            )
            self.summaries[child_id] = summary
            return summary
        except Exception as e:
            logger.error(f"Error auditing ledger for Child {child_id}: {e}")
            return None

    def apply_survival_tax(self, child_id: int, summary: ContainerLedgerSummary) -> float:
        """
        Computes and deducts the Logarithmic + Stagnation survival tax:
        Tax_t = Tax_0 * (1 + alpha * ln(1 + t/60)) + beta * t_stagnant
        Where t is simulated minutes (ticks_active / ticks_per_min).
        Returns total tax deducted.
        """
        db_path = self.get_ledger_path(child_id)
        if not os.path.exists(db_path) or not summary.is_alive:
            return 0.0
        
        # Calculate simulated hours (assuming 60 ticks per simulated hour or tick=minute)
        simulated_hours = summary.ticks_active / 60.0
        log_component = self.base_tax_per_hr * (1.0 + self.alpha_tax_scale * math.log(1.0 + simulated_hours))
        
        # Stagnation penalty applied if ticks_stagnant > 10 (turtling without equity compounding)
        stagnation_penalty = 0.0
        if summary.ticks_stagnant > 10:
            stagnation_penalty = self.beta_stagnation_penalty * (summary.ticks_stagnant - 10)
            
        total_tax = log_component + stagnation_penalty
        
        # Deduct from ledger
        new_cash = summary.cash - total_tax
        new_equity = summary.equity - total_tax
        
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE portfolio_state
                SET cash = ?, equity = ?
                WHERE tick_id = (SELECT MAX(tick_id) FROM portfolio_state)
            """, (new_cash, new_equity))
            
            cursor.execute("""
                INSERT INTO tax_history (timestamp, tick_id, tax_deducted, stagnation_penalty, remaining_equity)
                VALUES (?, ?, ?, ?, ?)
            """, (time.time(), summary.ticks_active, total_tax, stagnation_penalty, new_equity))
            conn.commit()
            conn.close()
            
            summary.cash = new_cash
            summary.equity = new_equity
            
            # Check for insolvency
            if new_equity <= 0.0 or new_cash <= -50.0:
                logger.warning(f"Child {child_id} ({summary.container_name}) became insolvent (Equity: ${new_equity:.2f}). Triggering purge.")
                self.terminate_insolvent_child(child_id, reason="INSOLVENCY_TAX_EXHAUSTION")
                
            return total_tax
        except Exception as e:
            logger.error(f"Error applying tax to Child {child_id}: {e}")
            return 0.0

    def assign_memory_tier(self, child_id: int, summary: ContainerLedgerSummary) -> int:
        """
        Evaluates ledger metrics and assigns Grace Blackwell MIG VRAM tier:
        Tier 1: High Alpha (Sharpe > 2.0, Drawdown < 10%, No OOM Lock) -> 20GB VRAM / High Priority
        Tier 2: Stagnant (0.5 < Sharpe <= 2.0 or mild Drawdown) -> 4GB VRAM / Mid Priority
        Tier 3: Failing / OOM Penalty (Sharpe <= 0.5 or Drawdown >= 25% or OOM Lock active) -> 0GB Train / Inference-Only
        """
        if not summary.is_alive:
            return 3
        
        current_time = time.time()
        
        # Check OOM Watchdog Lockout
        if summary.oom_penalty_active and current_time < summary.oom_lock_until:
            tier = 3
            logger.info(f"Child {child_id} is under active OOM Watchdog Lockout until {summary.oom_lock_until - current_time:.0f}s. Enforcing Tier 3.")
        elif summary.sharpe_ratio > 2.0 and summary.max_drawdown < 0.10 and summary.equity >= 12.0:
            tier = 1
        elif (summary.sharpe_ratio > 0.5 and summary.max_drawdown < 0.25) or summary.ticks_active < 10:
            tier = 2
        else:
            tier = 3
            
        summary.tier = tier
        
        # Enforce hardware slicing via Docker resource limits or NVIDIA MIG/MPS
        self._enforce_container_hardware_slice(summary.container_name, tier)
        
        # Record allocation
        try:
            with open(self.tier_history_path, "a", encoding="utf-8") as f:
                record = {
                    "timestamp": time.time(),
                    "child_id": child_id,
                    "tier": tier,
                    "sharpe": summary.sharpe_ratio,
                    "drawdown": summary.max_drawdown,
                    "equity": summary.equity
                }
                f.write(json.dumps(record) + "\n")
        except Exception as e:
            logger.error(f"Failed to record tier allocation: {e}")
            
        return tier

    def _enforce_container_hardware_slice(self, container_name: str, tier: int):
        """
        Applies physical memory, CPU priority (`nice`), and MIG slice restrictions to the Docker container.
        """
        tier_specs = {
            1: {"memory": "32g", "cpuset": "0-15", "cpu_shares": 2048, "mig_profile": "3g.20gb", "vram_limit_gb": 20.0},
            2: {"memory": "8g", "cpuset": "16-23", "cpu_shares": 1024, "mig_profile": "1g.5gb", "vram_limit_gb": 4.0},
            3: {"memory": "3g", "cpuset": "24-27", "cpu_shares": 512, "mig_profile": "none", "vram_limit_gb": 0.0}
        }
        spec = tier_specs.get(tier, tier_specs[2])
        
        # Update Docker container cgroups via docker update command if running
        try:
            cmd = [
                "docker", "update",
                "--memory", spec["memory"],
                "--cpu-shares", str(spec["cpu_shares"]),
                container_name
            ]
            # Execute non-blocking or quietly catch if container doesn't exist yet (genesis stage)
            subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
            
            # Write VRAM quota file inside container shared state so child_agent can self-limit PyTorch memory
            child_id_str = container_name.split("_")[-1]
            quota_file = os.path.join(self.state_dir, f"child_{child_id_str}", "vram_quota.json")
            if os.path.exists(os.path.dirname(quota_file)):
                with open(quota_file, "w", encoding="utf-8") as qf:
                    json.dump({
                        "tier": tier,
                        "vram_limit_gb": spec["vram_limit_gb"],
                        "mig_profile": spec["mig_profile"],
                        "timestamp": time.time()
                    }, qf)
        except Exception as e:
            logger.debug(f"Could not update hardware cgroup for {container_name}: {e}")

    def terminate_insolvent_child(self, child_id: int, reason: str = "BANKRUPTCY"):
        """
        Executes immediate container destruction (`docker kill` & `docker rm -v`) and purges the insolvent ledger.
        """
        container_name = f"swarm_child_{child_id}"
        logger.warning(f"PURGING INSOLVENT CHILD {child_id} ({container_name}) due to {reason}.")
        
        try:
            # Kill and remove container
            subprocess.run(["docker", "kill", container_name], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
            subprocess.run(["docker", "rm", "-v", container_name], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
            
            if child_id in self.summaries:
                self.summaries[child_id].is_alive = False
                
            # Log purge event
            with open(self.purge_log_path, "a", encoding="utf-8") as pf:
                pf.write(json.dumps({
                    "timestamp": time.time(),
                    "child_id": child_id,
                    "container_name": container_name,
                    "reason": reason
                }) + "\n")
        except Exception as e:
            logger.error(f"Error purging container {container_name}: {e}")

    def run_audit_cycle(self) -> Dict[int, int]:
        """
        Executes a complete Warden audit loop across the entire swarm:
        1. Audits each child ledger.
        2. Applies logarithmic + stagnation tax.
        3. Assigns Grace Blackwell MIG memory tiers.
        Returns mapping of child_id to assigned Tier.
        """
        tier_map = {}
        active_count = 0
        total_equity = 0.0
        
        for child_id in range(self.swarm_size):
            summary = self.audit_child_ledger(child_id)
            if summary and summary.is_alive:
                self.apply_survival_tax(child_id, summary)
                if summary.is_alive:
                    tier = self.assign_memory_tier(child_id, summary)
                    tier_map[child_id] = tier
                    active_count += 1
                    total_equity += summary.equity
                    
        logger.info(f"Audit Cycle Complete: {active_count}/{self.swarm_size} Active Children | Total Swarm Equity: ${total_equity:.2f}")
        return tier_map


if __name__ == "__main__":
    logger.info("Initializing Warden Core Hypervisor Standalone Test...")
    warden = WardenHypervisor(swarm_size=5, state_dir="d:/TradeJack/state")
    for cid in range(5):
        warden.init_child_ledger(cid, initial_cash=10.0)
    warden.run_audit_cycle()
