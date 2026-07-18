"""
Sovereign Child Agent (`SovereignChild` shell adapting `automaton` and `Stock-Prediction-Models`).
Executes continuous `Think -> Act -> Observe` loops inside isolated Grace Blackwell containers.
Petitions Warden for compute, trades across the physical LOB environment, tracks portfolio state in SQLite,
self-modifies architectures under memory pressure, tags HWM checkpoints, and executes instant rollbacks upon drawdown.
"""

import os
import sys
import time
import json
import logging
import random
import numpy as np
from typing import Dict, Any, List, Optional, Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] (SovereignChild) %(message)s")
logger = logging.getLogger("SovereignChild")

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from physics.lob_env import TradeJackLOBEnv
from physics.portfolio_tracker import PortfolioAccountingEngine
from swarm.skills.dgx_compute_skill import DGXComputeSkill
from swarm.self_mod_manager import SelfModEngine
from swarm.git_rollback import GitFinancialRollback
from swarm.social_relay import SocialRelayBridge
from swarm.rl_mechanics import HindsightExperienceReplay, PopulationBasedTrainingEngine, AdversarialGANSpoofer


class SovereignChild:
    """
    Sovereign AI financial organism running inside Docker (or local testing thread).
    """

    def __init__(
        self,
        child_id: int = 0,
        symbol: str = "BTC-USDT",
        initial_cash: float = 10.0,
        state_dir: str = "d:/TradeJack/state",
        data_store_dir: str = "d:/TradeJack/data_store"
    ):
        self.child_id = child_id
        self.symbol = symbol
        self.initial_cash = initial_cash
        self.state_dir = os.path.abspath(state_dir)
        self.data_store_dir = os.path.abspath(data_store_dir)
        
        # Initialize Subsystems
        self.compute_skill = DGXComputeSkill(child_id=self.child_id)
        self.self_mod = SelfModEngine(child_id=self.child_id, state_dir=self.state_dir)
        self.rollback_engine = GitFinancialRollback(child_id=self.child_id, repo_dir="d:/TradeJack")
        self.portfolio_engine = PortfolioAccountingEngine(child_id=self.child_id, state_dir=self.state_dir, initial_cash=self.initial_cash)
        self.social_relay = SocialRelayBridge(child_id=self.child_id, state_dir=self.state_dir)
        self.her_buffer = HindsightExperienceReplay(capacity=10000)
        self.spoofer = AdversarialGANSpoofer(spoof_intensity=0.2)
        
        # Physical Environment
        self.env = TradeJackLOBEnv(
            symbol=self.symbol,
            initial_cash=self.initial_cash,
            data_store_dir=self.data_store_dir,
            child_id=self.child_id
        )
        
        self.current_tier = 2
        self.vram_limit_gb = 4.0
        self.survival_mode = "NORMAL"
        self.is_terminated = False

    def petition_and_adapt(self):
        """Checks with Warden and updates tier assignment."""
        petition = self.compute_skill.petition_for_vram(requested_vram_gb=8.0, reason="Routine model forward pass and EWC adaptation")
        if isinstance(petition, dict) and "tier" in petition:
            self.current_tier = int(petition["tier"])
            self.vram_limit_gb = float(petition.get("vram_limit_gb", 4.0))
            logger.info(f"Child {self.child_id} assigned Tier {self.current_tier} ({self.vram_limit_gb}GB VRAM).")
            # Adapt model architecture if needed
            self.self_mod.swap_active_architecture(
                target_model_name=self.self_mod.active_model.model_name,
                current_tier=self.current_tier
            )

    def check_and_enforce_survival_mode(self, portfolio_summary: Dict[str, Any], step: int) -> str:
        """
        Monitors live cash/equity and enforces survival mode transitions (`HIGH`, `NORMAL`, `LOW_COMPUTE`, `CRITICAL`).
        Adapted from Conway-Research/automaton (`low-compute.ts` & `monitor.ts`).
        """
        eq = portfolio_summary.get("equity", self.portfolio_engine.equity)
        cash = portfolio_summary.get("cash", self.portfolio_engine.cash)
        
        old_mode = self.survival_mode
        if eq < 3.0 or cash < 0.0:
            self.survival_mode = "CRITICAL"
        elif eq < 5.0 or cash < 5.0:
            self.survival_mode = "LOW_COMPUTE"
        elif eq > 20.0 and cash > 20.0:
            self.survival_mode = "HIGH"
        else:
            self.survival_mode = "NORMAL"
            
        if old_mode != self.survival_mode:
            logger.warning(f"Child {self.child_id} transitioned Survival Mode: {old_mode} -> {self.survival_mode} (Eq: ${eq:.2f}, Cash: ${cash:.2f})")
            if self.survival_mode in ["LOW_COMPUTE", "CRITICAL"]:
                # Switch to lightweight scalping model to conserve compute VRAM and survival tax
                if self.self_mod.active_model.model_name != "Deep-Q-learning":
                    logger.info(f"Survival Mode '{self.survival_mode}' forced downgrade to 'Deep-Q-learning' scalper.")
                    self.self_mod.swap_active_architecture("Deep-Q-learning", current_tier=3)
                # Emergency P2P weight petition
                peers = self.social_relay.query_top_peers(min_sharpe=1.0)
                if peers:
                    best_peer = peers[0]
                    target_lineage = best_peer["lineage_id"]
                    logger.info(f"Emergency Survival Petition: requesting peer weights '{target_lineage}' via Escrow...")
                    self.social_relay.request_peer_weights_via_escrow(target_lineage, offered_usdc=0.25)
        return self.survival_mode

    def think(self, obs: Dict[str, np.ndarray]) -> float:
        """
        Runs neural model inference across rolling LOB window to produce position allocation [-1.0, 1.0].
        """
        lob_seq = obs["lob_sequence"]
        # Inject adversarial spoof noise during training/adaptation phase for resilience
        spoofed_lob = self.spoofer.inject_spoof_noise(lob_seq)
        
        if TORCH_AVAILABLE and hasattr(self.self_mod.active_model, "net"):
            t_in = torch.from_numpy(spoofed_lob).unsqueeze(0).to(torch.float32)
            with torch.no_grad():
                out = self.self_mod.active_model.forward(t_in)
                if isinstance(out, torch.Tensor):
                    action_val = float(out.mean().cpu().numpy())
                else:
                    action_val = float(np.mean(out))
        else:
            # Numpy / simulation model evaluation
            out = self.self_mod.active_model.forward(spoofed_lob)
            action_val = float(np.mean(out))
            
        # Scale and clip action to [-1.0, 1.0]
        return float(np.clip(action_val, -1.0, 1.0))

    def run_crucible_loop(self, max_steps: int = 500) -> Dict[str, Any]:
        """
        Main autonomous survival loop inside the container.
        """
        logger.info(f"SovereignChild {self.child_id} initiating Crucible Loop (Symbol: {self.symbol}, Cash: ${self.initial_cash:.2f})...")
        self.petition_and_adapt()
        
        obs, info = self.env.reset()
        summary = {}
        
        for step in range(max_steps):
            # 1. Think
            action = self.think(obs)
            
            # 2. Act
            next_obs, reward, terminated, truncated, env_info = self.env.step([action])
            
            # 3. Record in SQLite Ledger
            portfolio_summary = self.portfolio_engine.record_step(
                new_cash=env_info["cash"],
                new_equity=env_info["equity"],
                tick_id=step
            )
            
            # Check survival mode transitions
            self.check_and_enforce_survival_mode(portfolio_summary, step)
            
            # Push transition to HER buffer
            self.her_buffer.push(
                state=obs["lob_sequence"],
                action=action,
                reward=reward,
                next_state=next_obs["lob_sequence"],
                achieved_equity=env_info["equity"],
                desired_equity=env_info.get("peak_equity", 10.0) * 1.1,
                done=terminated
            )
            
            # 4. Observe & Reflect (HWM Checkpoint / Rollback / Social Relay)
            current_eq = env_info["equity"]
            current_dd = env_info["max_drawdown"]
            
            # Check HWM tagging
            tag = self.rollback_engine.check_and_checkpoint(current_equity=current_eq)
            if tag and portfolio_summary["sharpe_ratio"] > 1.5:
                # Broadcast high-Sharpe weights to social relay
                weights_path = os.path.join(self.state_dir, f"child_{self.child_id}", f"weights_{tag}.pt")
                self.social_relay.broadcast_market_insight(
                    equity=current_eq,
                    sharpe_ratio=portfolio_summary["sharpe_ratio"],
                    state_dict_path=weights_path,
                    description=f"HWM {tag} on {self.symbol} (Model: {self.self_mod.active_model.model_name})"
                )
                
            # Check Drawdown Breach Rollback (>15%)
            did_rollback = self.rollback_engine.execute_rollback_if_breached(
                current_equity=current_eq,
                current_drawdown=current_dd
            )
            if did_rollback:
                logger.warning(f"Child {self.child_id} reverted after drawdown breach at step {step}.")
                # Reset environment equity tracking to restored state if live
                
            # Check Stagnation ("Turtling") -> Request P2P weights or tier swap
            if portfolio_summary["ticks_stagnant"] >= 30 and step % 30 == 0:
                logger.info(f"Child {self.child_id} stagnant for 30 ticks. Querying Social Relay for peer breakthrough...")
                peers = self.social_relay.query_top_peers(min_sharpe=1.2)
                if peers:
                    best_peer = peers[0]
                    target_lineage = best_peer["lineage_id"]
                    self.social_relay.request_peer_weights_via_escrow(target_lineage, offered_usdc=0.50)
                    
            obs = next_obs
            
            if terminated or truncated:
                logger.info(f"Child {self.child_id} loop concluded at step {step}. Equity: ${current_eq:.2f}, Sharpe: {portfolio_summary['sharpe_ratio']:.2f}.")
                break
                
        self.is_terminated = True
        return {
            "child_id": self.child_id,
            "steps_completed": step + 1,
            "final_equity": self.portfolio_engine.equity,
            "peak_equity": self.portfolio_engine.peak_equity,
            "max_drawdown": self.portfolio_engine.max_drawdown,
            "sharpe_ratio": self.portfolio_engine.compute_risk_adjusted_ratios()[0],
            "sortino_ratio": self.portfolio_engine.compute_risk_adjusted_ratios()[1],
            "active_tier": self.current_tier,
            "active_model": self.self_mod.active_model.model_name
        }


if __name__ == "__main__":
    logger.info("Testing SovereignChild standalone execution...")
    # Ensure synthetic data exists
    from data_forge.parquet_ingest import ParquetIngestPipeline
    import asyncio
    ingest = ParquetIngestPipeline(data_store_dir="d:/TradeJack/data_store")
    asyncio.run(ingest.generate_synthetic_crucible_data(symbol="BTC-USDT", num_days=1, ticks_per_day=150))
    
    child = SovereignChild(child_id=1, symbol="BTC-USDT", initial_cash=10.0)
    result = child.run_crucible_loop(max_steps=25)
    print("Crucible Loop Final Summary:", json.dumps(result, indent=2))
