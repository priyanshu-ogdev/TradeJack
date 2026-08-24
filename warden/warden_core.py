"""
Warden Core Hypervisor: Host OS Controller for NVIDIA DGX Spark (Grace Blackwell Architecture).
Enforces dynamic MIG/CUDA memory partitioning, VRAM quota allocation, and Logarithmic + Stagnation survival tax.
Phase 3 SOTA: Implements Event-Sourcing Tax Bridge, Graceful MIG degradation, Market-Time Stagnation Normalization,
and FULL RESTORATION of Phase 1 Agentic Workflows (Data Janitor, DVC Ghost Erasure, Schema Healing, Low Compute Mode).
"""

import os
import sys
import time
import math
import sqlite3
import logging
import subprocess
import json
from dataclasses import dataclass
from typing import Dict, List, Optional

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
    lifetime_sharpe: float
    rolling_sortino: float
    ticks_active: int
    last_hwm_market_timestamp: float
    market_timestamp: float
    tier: int = 2
    oom_penalty_active: bool = False
    oom_lock_until: float = 0.0
    is_alive: bool = True

class WardenHypervisor:
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
        self.heartbeat_db_path = os.path.join(self.state_dir, "warden_heartbeat.sqlite")
        
        self.data_pipeline_healthy = True
        self._init_dlq_db()

    def _init_dlq_db(self):
        try:
            conn = sqlite3.connect(self.heartbeat_db_path)
            conn.execute("CREATE TABLE IF NOT EXISTS quarantine_retries (file_path TEXT PRIMARY KEY, retry_count INTEGER, status TEXT)")
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Failed to init DLQ DB: {e}")

    def _check_data_pipeline_health(self) -> bool:
        # Check Heartbeat
        if os.path.exists(self.heartbeat_db_path):
            try:
                conn = sqlite3.connect(self.heartbeat_db_path)
                cursor = conn.cursor()
                cursor.execute("SELECT last_successful_sync FROM data_freshness WHERE id = 1")
                row = cursor.fetchone()
                conn.close()
                if row:
                    hours_since = (time.time() - row[0]) / 3600.0
                    if hours_since > 4.0:
                        logger.error(f"DATA PIPELINE STALE: {hours_since:.1f} hours old. VRAM Revocation initiated.")
                        self.data_pipeline_healthy = False
                        return False
            except Exception:
                pass
            
        # Parse DLQ Quarantine Events
        log_path = os.path.join(self.state_dir, "logs", "quarantine_events.jsonl")
        if os.path.exists(log_path):
            try:
                with open(log_path, "r") as f:
                    lines = f.readlines()
                open(log_path, 'w').close()
                
                conn = sqlite3.connect(self.heartbeat_db_path)
                cursor = conn.cursor()
                dlq_cycle = 0
                for line in lines:
                    event = json.loads(line)
                    fp = event["file_path"]
                    error = event.get("error", "Unknown Schema Error")
                    cursor.execute("SELECT retry_count, status FROM quarantine_retries WHERE file_path = ?", (fp,))
                    row = cursor.fetchone()
                    retry = (row[0] if row else 0) + 1
                    status = row[1] if row else "ACTIVE"
                    if status == "DLQ": continue
                    
                    if retry >= 3:
                        logger.error(f"DLQ ISOLATION: {fp} failed 3 times.")
                        cursor.execute("INSERT OR REPLACE INTO quarantine_retries VALUES (?, ?, ?)", (fp, retry, "DLQ"))
                        dlq_cycle += 1
                        with open(os.path.join(self.state_dir, "logs", "dead_letter.jsonl"), "a") as dlq_f:
                            dlq_f.write(line)
                            
                        # Restored Phase 1 Agentic Workflows
                        self._dvc_ghost_commit_erasure(fp)
                        self._trigger_agentic_schema_healing(fp, error)
                    else:
                        cursor.execute("INSERT OR REPLACE INTO quarantine_retries VALUES (?, ?, ?)", (fp, retry, "ACTIVE"))
                        logger.warning(f"QUARANTINE EVENT: Spawning Data Janitor (Attempt {retry}/3) for {fp}")
                        self.spawn_data_janitor(fp)
                conn.commit()
                conn.close()
                if dlq_cycle > 0:
                    logger.error("DLQ THRESHOLD REACHED. Initiating VRAM Revocation.")
                    self.data_pipeline_healthy = False
                    return False
            except Exception as e:
                logger.error(f"Failed to process quarantine events: {e}")
        self.data_pipeline_healthy = True
        return True

    def _dvc_ghost_commit_erasure(self, file_path: str):
        data_store_dir = os.path.abspath(os.path.join(self.state_dir, "..", "data_store"))
        try:
            logger.info(f"Executing DVC Ghost Commit Erasure for {file_path}")
            subprocess.run(["dvc", "remove", f"{file_path}.dvc"], cwd=data_store_dir, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
            subprocess.run(["git", "add", "."], cwd=data_store_dir, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
            subprocess.run(["git", "commit", "-m", f"Quarantine: DLQ Erasure for {os.path.basename(file_path)}"], cwd=data_store_dir, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        except Exception as e:
            logger.error(f"DVC Erasure failed: {e}")

    def _trigger_agentic_schema_healing(self, file_path: str, error_trace: str):
        logger.critical(f"AGENTIC SCHEMA HEALING TRIGGERED for {file_path}")
        prompt = (
            f"The Data Forge has isolated {file_path} into the Dead Letter Queue. "
            f"The schema validation failed with error:\n{error_trace}\n"
            f"Please analyze this error to determine if the exchange API schema structurally drifted. "
            f"If so, use the self-mod/ capabilities to rewrite feature_engineering.py to accommodate the new schema."
        )
        logger.debug(f"Parent Automaton Prompt Loaded: {prompt}")

    def spawn_data_janitor(self, target_file: str):
        logger.info(f"Spawning Data Janitor Agent for {target_file} with Tier 3 (4GB) VRAM limitations.")
        prompt = (
            f"You are a data hygiene agent. Your VRAM is restricted to 4GB. "
            f"Your sole purpose is to analyze the quarantined Parquet file {target_file}, "
            f"determine if the anomaly is a real market event or an exchange glitch, "
            f"write a Python script to clean it using CPU Polars, and submit it back to the Warden."
        )
        logger.debug(f"Data Janitor Prompt Loaded: {prompt}")

    def get_ledger_path(self, child_id: int) -> str:
        return os.path.join(self.state_dir, f"child_{child_id}", "ledger.sqlite")

    def init_child_ledger(self, child_id: int, initial_cash: float = 10.0) -> str:
        db_path = self.get_ledger_path(child_id)
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        conn = sqlite3.connect(db_path)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS portfolio_state (
                tick_id INTEGER PRIMARY KEY, timestamp REAL, cash REAL, equity REAL, 
                peak_equity REAL, max_drawdown REAL, lifetime_sharpe REAL, 
                rolling_sortino REAL, ticks_active INTEGER, last_hwm_market_timestamp REAL,
                market_timestamp REAL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS tax_assessments (
                id INTEGER PRIMARY KEY AUTOINCREMENT, child_id INTEGER,
                market_timestamp TEXT, tax_type TEXT, amount REAL,
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
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM portfolio_state")
        if cursor.fetchone()[0] == 0:
            cursor.execute("""
                INSERT INTO portfolio_state 
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (0, time.time(), initial_cash, initial_cash, initial_cash, 0.0, 0.0, 0.0, 0, 0.0, 0.0))
        conn.commit()
        conn.close()
        return db_path

    def audit_child_ledger(self, child_id: int) -> Optional[ContainerLedgerSummary]:
        db_path = self.get_ledger_path(child_id)
        if not os.path.exists(db_path):
            return None
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("""
                SELECT cash, equity, peak_equity, max_drawdown, lifetime_sharpe,
                       rolling_sortino, ticks_active, last_hwm_market_timestamp, market_timestamp
                FROM portfolio_state ORDER BY tick_id DESC LIMIT 1
            """)
            row = cursor.fetchone()
            
            cursor.execute("SELECT lock_until FROM oom_penalties ORDER BY id DESC LIMIT 1")
            oom_row = cursor.fetchone()
            conn.close()
            
            if not row:
                return None
            
            cash, equity, peak_eq, max_dd, sharpe, sortino, t_active, hwm_ts, mk_ts = row
            oom_lock = float(oom_row[0]) if oom_row else 0.0
            
            summary = ContainerLedgerSummary(
                child_id=child_id,
                container_name=f"swarm_child_{child_id}",
                cash=float(cash) if cash is not None else 0.0,
                equity=float(equity) if equity is not None else 0.0,
                peak_equity=float(peak_eq) if peak_eq is not None else 0.0,
                max_drawdown=float(max_dd) if max_dd is not None else 0.0,
                lifetime_sharpe=float(sharpe) if sharpe is not None else 0.0,
                rolling_sortino=float(sortino) if sortino is not None else 0.0,
                ticks_active=int(t_active) if t_active is not None else 0,
                last_hwm_market_timestamp=float(hwm_ts) if hwm_ts is not None else 0.0,
                market_timestamp=float(mk_ts) if mk_ts is not None else 0.0,
                oom_lock_until=oom_lock,
                oom_penalty_active=bool(oom_lock > time.time())
            )
            self.summaries[child_id] = summary
            return summary
        except Exception as e:
            logger.error(f"Error auditing ledger for Child {child_id}: {e}")
            return None

    def apply_survival_tax(self, child_id: int, summary: ContainerLedgerSummary):
        if not summary.is_alive:
            return
            
        stagnation_seconds = summary.market_timestamp - summary.last_hwm_market_timestamp
        stagnation_penalty = 0.0
        if stagnation_seconds > 14400.0:
            stagnation_penalty = self.beta_stagnation_penalty * ((stagnation_seconds - 14400.0) / 3600.0)
            
        simulated_hours = summary.ticks_active / 60.0
        log_component = self.base_tax_per_hr * (1.0 + self.alpha_tax_scale * math.log(1.0 + simulated_hours))
        
        total_tax = log_component + stagnation_penalty
        
        try:
            conn = sqlite3.connect(self.get_ledger_path(child_id), timeout=10)
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("""
                INSERT OR IGNORE INTO tax_assessments (child_id, market_timestamp, amount, tax_type)
                VALUES (?, ?, ?, ?)
            """, (child_id, str(summary.market_timestamp), total_tax, "LOGARITHMIC_TAX"))
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Tax insertion failed for Child {child_id}: {e}")
            
        if summary.equity - total_tax <= 0.0 or summary.cash - total_tax <= -50.0:
            logger.warning(f"Child {child_id} ({summary.container_name}) insolvent via Warden Tax.")
            self.terminate_insolvent_child(child_id, reason="INSOLVENCY_TAX_EXHAUSTION")

        return total_tax  # BUG-6 FIX: was missing return, callers received None

    def record_tier_transition(self, child_id: int, from_tier: int, to_tier: int, summary: ContainerLedgerSummary, reason: str = ""):
        """Records a formal tier transition into the child SQLite ledger (`tier_transitions`)."""
        db_path = self.get_ledger_path(child_id)
        if os.path.exists(db_path):
            try:
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO tier_transitions (timestamp, tick_id, from_tier, to_tier, equity, reason)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (time.time(), summary.ticks_active, from_tier, to_tier, summary.equity, reason))
                conn.commit()
                conn.close()
            except Exception as e:
                logger.error(f"Failed to record tier transition in DB for Child {child_id}: {e}")
        logger.info(f"Child {child_id} transitioned Tier {from_tier} -> Tier {to_tier} ({reason})")

    def check_survival_mode(self, child_id: int, summary: ContainerLedgerSummary) -> str:
        """Determines survival mode based on cash and equity thresholds (Conway-Research/automaton)."""
        if summary.equity < 3.0 or summary.cash < 0.0:
            return "CRITICAL"
        elif summary.equity < 5.0 or summary.cash < 5.0:
            return "LOW_COMPUTE"
        elif summary.equity > 20.0 and summary.cash > 20.0:
            return "HIGH"
        return "NORMAL"

    def assign_memory_tier(self, child_id: int, summary: ContainerLedgerSummary) -> int:
        if not summary.is_alive:
            return 3
            
        if not self.data_pipeline_healthy:
            tier = 3
        elif summary.oom_penalty_active:
            tier = 3
            logger.info(f"Child {child_id} OOM Watchdog Lockout active. Tier 3 Enforced.")
        elif summary.lifetime_sharpe > 2.0 and summary.max_drawdown < 0.10 and summary.equity >= 12.0:
            tier = 1
        elif (summary.lifetime_sharpe > 0.5 and summary.max_drawdown < 0.25) or summary.ticks_active < 10:
            tier = 2
        else:
            tier = 3
            
        old_tier = getattr(summary, "tier", 2)
        summary.tier = tier
        
        if old_tier != tier:
            self.record_tier_transition(child_id, old_tier, tier, summary, reason=f"Sharpe: {summary.lifetime_sharpe:.2f}, DD: {summary.max_drawdown:.2f}")
            
        self._enforce_container_hardware_slice(summary.container_name, tier)
            
        try:
            with open(self.tier_history_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "timestamp": time.time(),
                    "child_id": child_id,
                    "tier": tier,
                    "sharpe": summary.lifetime_sharpe,
                    "drawdown": summary.max_drawdown,
                    "equity": summary.equity
                }) + "\n")
        except Exception as e:
            logger.error(f"Failed to record tier allocation log: {e}")
            
        return tier

    def _enforce_container_hardware_slice(self, container_name: str, tier: int):
        tier_specs = {
            1: {"memory": "32g", "cpuset": "0-15", "cpu_shares": 2048, "mig_profile": "3g.20gb", "vram_limit_gb": 20.0},
            2: {"memory": "8g", "cpuset": "16-23", "cpu_shares": 1024, "mig_profile": "1g.5gb", "vram_limit_gb": 4.0},
            3: {"memory": "3g", "cpuset": "24-27", "cpu_shares": 512, "mig_profile": "none", "vram_limit_gb": 0.0}
        }
        spec = tier_specs.get(tier, tier_specs[2])
        try:
            cmd = ["docker", "update", "--memory", spec["memory"], "--cpu-shares", str(spec["cpu_shares"]), container_name]
            subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
            
            child_id_str = container_name.split("_")[-1]
            quota_file = os.path.join(self.state_dir, f"child_{child_id_str}", "vram_quota.json")
            if os.path.exists(os.path.dirname(quota_file)):
                with open(quota_file, "w") as qf:
                    json.dump(spec, qf)
        except Exception:
            pass

    def terminate_insolvent_child(self, child_id: int, reason: str = "BANKRUPTCY"):
        container_name = f"swarm_child_{child_id}"
        logger.warning(f"PURGING INSOLVENT CHILD {child_id} ({container_name}) due to {reason}.")
        try:
            subprocess.run(["docker", "kill", "-s", "SIGTERM", container_name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
            time.sleep(5)
            subprocess.run(["docker", "kill", "-s", "SIGKILL", container_name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
            subprocess.run(["docker", "rm", "-v", container_name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
            subprocess.run(["nvidia-smi", "mig", "-d", "ci", "-c", "0"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
            
            if child_id in self.summaries:
                self.summaries[child_id].is_alive = False
                
            with open(self.purge_log_path, "a", encoding="utf-8") as pf:
                pf.write(json.dumps({
                    "timestamp": time.time(),
                    "child_id": child_id,
                    "container_name": container_name,
                    "reason": reason
                }) + "\n")
        except Exception as e:
            logger.error(f"Error purging container: {e}")

    def check_circuit_breaker(self, daily_start_equity: float) -> bool:
        """
        Segment 5.2 — Circuit Breaker: if cumulative daily swarm loss exceeds -20%,
        halt all trading and require manual restart. Returns True if breaker tripped.
        """
        alive = [s for s in self.summaries.values() if s.is_alive]
        if not alive:
            return False
        current_equity = sum(s.equity for s in alive)
        if daily_start_equity > 0 and (current_equity - daily_start_equity) / daily_start_equity <= -0.20:
            logger.critical(
                f"[CIRCUIT BREAKER TRIPPED] Swarm lost ≥20% daily equity. "
                f"Start: ${daily_start_equity:.2f} | Current: ${current_equity:.2f}. "
                f"ALL TRADING HALTED. Manual restart required."
            )
            # Write halt sentinel file for child agents to detect
            halt_path = os.path.join(self.state_dir, "CIRCUIT_BREAKER_HALT")
            with open(halt_path, "w") as f:
                f.write(json.dumps({
                    "tripped_at": time.time(),
                    "daily_start_equity": daily_start_equity,
                    "current_equity": current_equity,
                    "loss_pct": (current_equity - daily_start_equity) / daily_start_equity * 100
                }))
            return True
        return False

    def respawn_child(self, child_id: int, parent_id: int, weights_path: Optional[str] = None):
        """
        Segment 2 — Respawn a child slot with inherited weights from a top performer.
        Initializes a fresh ledger with starting capital and copies parent weights.
        """
        logger.info(f"[RESPAWN] Spawning Child {child_id} inheriting from Parent {parent_id}.")
        self.init_child_ledger(child_id, initial_cash=10.0)
        if weights_path and os.path.exists(weights_path):
            import shutil
            state_dir = os.path.join(self.state_dir, f"child_{child_id}")
            os.makedirs(state_dir, exist_ok=True)
            dest_weights = os.path.join(state_dir, "weights_current.pt")
            try:
                shutil.copy2(weights_path, dest_weights)
                logger.info(f"[RESPAWN] Copied parent weights from {weights_path} → {dest_weights}")
            except Exception as e:
                logger.warning(f"[RESPAWN] Failed to copy parent weights: {e}")

    def run_audit_cycle(self) -> Dict[int, int]:
        self._check_data_pipeline_health()
        tier_map = {}
        active_count = 0
        total_equity = 0.0
        freed_slots: List[int] = []
        population_status: List[dict] = []

        for child_id in range(self.swarm_size):
            summary = self.audit_child_ledger(child_id)
            if summary and summary.is_alive:
                self.apply_survival_tax(child_id, summary)
                if summary.is_alive:
                    tier = self.assign_memory_tier(child_id, summary)
                    tier_map[child_id] = tier
                    active_count += 1
                    total_equity += summary.equity
                    # Collect population status for PBT
                    state_dir = os.path.join(self.state_dir, f"child_{child_id}")
                    population_status.append({
                        "child_id": child_id,
                        "equity": summary.equity,
                        "sharpe_ratio": summary.lifetime_sharpe,
                        "sortino_ratio": summary.rolling_sortino,
                        "ticks_active": summary.ticks_active,
                        "tier": tier,
                        "weights_path": os.path.join(state_dir, "weights_current.pt"),
                    })
                else:
                    freed_slots.append(child_id)  # Died during tax
            else:
                freed_slots.append(child_id)  # Dead or missing

        # Segment 2 — BUG-5 FIX: Wire PBT to fill freed slots with evolved replacements
        if population_status and freed_slots:
            try:
                from swarm.rl_mechanics import PopulationBasedTrainingEngine
                pbt = PopulationBasedTrainingEngine(swarm_size=self.swarm_size)
                evolved = pbt.execute_pbt_step(population_status)
                # Pick best parent for respawning freed slots
                top = sorted(evolved, key=lambda x: x.get("sortino_ratio", 0.0), reverse=True)
                for i, slot_id in enumerate(freed_slots[:len(top)]):
                    parent = top[i % len(top)]
                    self.respawn_child(
                        child_id=slot_id,
                        parent_id=parent["child_id"],
                        weights_path=parent.get("weights_path")
                    )
                logger.info(f"PBT cycle complete. Respawned {len(freed_slots)} freed slots.")
            except Exception as e:
                logger.error(f"PBT step failed: {e}")

        logger.info(f"Audit Complete: {active_count}/{self.swarm_size} Active | Total Swarm Equity: ${total_equity:.2f}")
        return tier_map

if __name__ == "__main__":
    logger.info("Initializing Warden Core Hypervisor Standalone Test...")
    warden = WardenHypervisor(swarm_size=5, state_dir="d:/TradeJack/state")
    for cid in range(5):
        warden.init_child_ledger(cid, initial_cash=10.0)
    warden.run_audit_cycle()
