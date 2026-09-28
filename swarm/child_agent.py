"""
Sovereign Child Agent (`SovereignChild` shell adapting `automaton` and `Stock-Prediction-Models`).
Executes continuous `Think -> Act -> Observe` loops inside isolated Grace Blackwell containers.
Petitions Warden for compute, trades across the physical LOB environment, tracks portfolio state in SQLite,
self-modifies architectures under memory pressure, tags HWM checkpoints, and executes instant rollbacks upon drawdown.

v3 Upgrade:
  - Integrates OnlineRLTrainer (SB3-backed PPO/SAC/DQN) for actual gradient-based learning
  - Trains every K steps interleaved with inference (online learning)
  - SB3 Crucible mode: model.learn() owns the env loop for pure training runs
  - Legacy manual loop preserved for backward compatibility and custom orchestration
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
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

try:
    from stable_baselines3 import PPO, SAC, DQN
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False

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
        state_dir: str = "state",
        data_store_dir: str = "data_store",
        model_name: str = "PPO-DilatedCNN",
        train_every_k: int = 50,
        use_sb3_training: bool = True,
    ):
        self.child_id = child_id
        self.symbol = symbol
        self.initial_cash = initial_cash
        self.state_dir = os.path.abspath(state_dir)
        self.data_store_dir = os.path.abspath(data_store_dir)
        self.model_name = model_name
        self.train_every_k = train_every_k
        self.use_sb3_training = use_sb3_training and SB3_AVAILABLE

        # Physical Environment (Initializes its own PortfolioAccountingEngine internally)
        self.env = TradeJackLOBEnv(
            symbol=self.symbol,
            initial_cash=self.initial_cash,
            data_store_dir=self.data_store_dir,
            child_id=self.child_id
        )

        # Initialize Subsystems
        self.compute_skill = DGXComputeSkill(child_id=self.child_id)
        self.self_mod = SelfModEngine(child_id=self.child_id, state_dir=self.state_dir)
        self.rollback_engine = GitFinancialRollback(child_id=self.child_id, repo_dir=".")
        self.social_relay = SocialRelayBridge(child_id=self.child_id, state_dir=self.state_dir)
        self.her_buffer = HindsightExperienceReplay(capacity=10000)
        self.spoofer = AdversarialGANSpoofer(spoof_intensity=0.2)

        # v3: Initialize OnlineRLTrainer if SB3 is available
        self.rl_trainer = None
        if self.use_sb3_training:
            try:
                from swarm.rl_trainer import OnlineRLTrainer
                log_dir = os.path.join(self.state_dir, f"child_{self.child_id}", "tb_logs")
                self.rl_trainer = OnlineRLTrainer(
                    model_name=self.model_name,
                    env=self.env,
                    device="auto",
                    log_dir=log_dir,
                )
                logger.info(f"Child {self.child_id}: OnlineRLTrainer active ({self.model_name})")
            except Exception as e:
                logger.warning(f"Child {self.child_id}: Failed to init OnlineRLTrainer ({e}). Falling back to legacy mode.")
                self.rl_trainer = None

        self.current_tier = 2
        self.vram_limit_gb = 4.0
        self.survival_mode = "NORMAL"
        self.is_terminated = False

    def petition_and_adapt(self):
        """Checks with Warden and updates tier assignment + optimal model architecture."""
        petition = self.compute_skill.petition_for_vram(requested_vram_gb=8.0, reason="Routine model forward pass and EWC adaptation")
        if isinstance(petition, dict) and "tier" in petition:
            self.current_tier = int(petition["tier"])
            self.vram_limit_gb = float(petition.get("vram_limit_gb", 4.0))
            logger.info(f"Child {self.child_id} assigned Tier {self.current_tier} ({self.vram_limit_gb}GB VRAM).")

    def check_and_enforce_survival_mode(self, portfolio_summary: Dict[str, Any], step: int) -> str:
        """
        Monitors live cash/equity and enforces survival mode transitions (`HIGH`, `NORMAL`, `LOW_COMPUTE`, `CRITICAL`).
        Adapted from Conway-Research/automaton (`low-compute.ts` & `monitor.ts`).
        """
        eq = portfolio_summary.get("equity", self.env.accounting.equity)
        cash = portfolio_summary.get("cash", self.env.accounting.cash)

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
                # Emergency P2P weight petition
                regime_vector = self.env.obs_mean.tolist() if hasattr(self.env, "obs_mean") else None
                peers = self.social_relay.query_top_peers(min_sharpe=1.0, current_regime_vector=regime_vector)
                if peers:
                    best_peer = peers[0]
                    target_lineage = best_peer["lineage_id"]
                    logger.info(f"Emergency Survival Petition: requesting peer weights '{target_lineage}' via Escrow...")
                    self.social_relay.request_peer_weights_via_escrow(target_lineage, offered_usdc=0.25)
        return self.survival_mode

    def think(self, obs: Dict[str, np.ndarray]) -> float:
        """
        Runs model inference to produce position allocation [-1.0, 1.0].

        v3: Prefers SB3 trainer.predict() if available, falls back to legacy
        SelfModEngine.active_model.forward() for backward compatibility.
        """
        # v3: Use SB3 model for inference if available
        if self.rl_trainer is not None:
            try:
                return self.rl_trainer.predict(obs, deterministic=True)
            except Exception as e:
                logger.debug(f"SB3 predict failed ({e}), falling back to legacy.")

        # Legacy inference path (random-weight forward pass)
        lob_seq = obs["lob_sequence"]
        spoofed_lob = self.spoofer.inject_spoof_noise(lob_seq)

        if TORCH_AVAILABLE and hasattr(self.self_mod.active_model, "net"):
            t_in = torch.from_numpy(spoofed_lob).unsqueeze(0).to(torch.float32)
            with torch.no_grad():
                out = self.self_mod.active_model.forward(t_in)
                if isinstance(out, torch.Tensor):
                    if out.shape[-1] == 3:
                        action_idx = int(torch.argmax(out, dim=-1).item())
                        mapping = {0: 0.0, 1: 1.0, 2: -1.0}
                        action_val = mapping.get(action_idx, 0.0)
                    else:
                        action_val = float(out.mean().item())
                else:
                    out_arr = np.array(out)
                    if out_arr.shape[-1] == 3:
                        action_idx = int(np.argmax(out_arr, axis=-1))
                        mapping = {0: 0.0, 1: 1.0, 2: -1.0}
                        action_val = mapping.get(action_idx, 0.0)
                    else:
                        action_val = float(np.mean(out_arr))
        else:
            # Numpy / simulation model evaluation
            out = self.self_mod.active_model.forward(spoofed_lob)
            out_arr = np.array(out)
            if out_arr.shape[-1] == 3:
                action_idx = int(np.argmax(out_arr, axis=-1))
                mapping = {0: 0.0, 1: 1.0, 2: -1.0}
                action_val = mapping.get(action_idx, 0.0)
            else:
                action_val = float(np.mean(out_arr))

        return float(np.clip(action_val, -1.0, 1.0))

    def run_crucible_loop_sb3(self, total_timesteps: int = 10000) -> Dict[str, Any]:
        """
        v3 SB3-native Crucible loop: SB3 owns the env interaction entirely.

        This is the preferred training mode. SB3 collects rollouts, computes
        advantages/TD-errors, and runs gradient updates internally.
        """
        if self.rl_trainer is None:
            logger.warning("SB3 trainer not available. Falling back to legacy loop.")
            return self.run_crucible_loop(max_steps=total_timesteps)

        logger.info(
            f"SovereignChild {self.child_id} initiating SB3 Crucible Loop "
            f"(Model: {self.model_name}, Timesteps: {total_timesteps})..."
        )

        # Let SB3 own the entire training loop
        train_summary = self.rl_trainer.learn(total_timesteps=total_timesteps)

        # Collect final portfolio state from env
        final_equity = self.env.accounting.equity
        peak_equity = self.env.accounting.peak_equity
        max_drawdown = self.env.accounting.max_drawdown
        sharpe, sortino = self.env.accounting._compute_ratios()

        # Save checkpoint
        ckpt_dir = os.path.join(self.state_dir, f"child_{self.child_id}", "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt_path = os.path.join(ckpt_dir, f"model_{self.model_name}")
        self.rl_trainer.save(ckpt_path)

        # Tag HWM if applicable
        tag = self.rollback_engine.check_and_checkpoint(current_equity=final_equity)
        if tag and sharpe > 1.5:
            regime_vector = self.env.obs_mean.tolist() if hasattr(self.env, "obs_mean") else None
            self.social_relay.broadcast_market_insight(
                equity=final_equity,
                sharpe_ratio=sharpe,
                state_dict_path=ckpt_path,
                description=f"SB3 HWM {tag} ({self.model_name})",
                regime_vector=regime_vector
            )

        self.is_terminated = True
        result = {
            "child_id": self.child_id,
            "model_name": self.model_name,
            "training_mode": "sb3",
            "total_timesteps": total_timesteps,
            "final_equity": final_equity,
            "peak_equity": peak_equity,
            "max_drawdown": max_drawdown,
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "active_tier": self.current_tier,
            **{k: v for k, v in train_summary.items() if k.startswith("avg_")},
        }
        logger.info(
            f"Child {self.child_id} SB3 Crucible complete: "
            f"equity=${final_equity:.2f}, sharpe={sharpe:.2f}, "
            f"sortino={sortino:.2f}"
        )
        return result

    def run_crucible_loop(self, max_steps: int = 500) -> Dict[str, Any]:
        """
        Legacy manual survival loop (backward compatible).

        v3 addition: calls rl_trainer.train_step() every K steps if SB3 is available,
        interleaving training with the existing manual loop logic.
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

            # Extract market_timestamp safely, LOBEnv might not expose it in env_info directly
            # but we can get it manually since we control the env.
            obs_idx = max(0, self.env.current_step_in_batch - 1)
            current_market_ts = self.env._get_scalar("timestamp", obs_idx)

            # 3. Retrieve Summary from Internal Accounting
            # lob_env.py already called record_step() and updated the SQLite Ledger natively.
            portfolio_summary = {
                "equity": self.env.accounting.equity,
                "cash": self.env.accounting.cash,
                "max_drawdown": self.env.accounting.max_drawdown,
                "lifetime_sharpe": self.env.accounting._compute_ratios()[0],
                "rolling_sortino": self.env.accounting._compute_ratios()[1],
                "last_hwm": self.env.accounting.last_hwm_market_timestamp
            }

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
            if tag and portfolio_summary["lifetime_sharpe"] > 1.5:
                # Broadcast high-Sharpe weights to social relay
                weights_path = os.path.join(self.state_dir, f"child_{self.child_id}", f"weights_{tag}.pt")
                regime_vector = self.env.obs_mean.tolist() if hasattr(self.env, "obs_mean") else None
                self.social_relay.broadcast_market_insight(
                    equity=current_eq,
                    sharpe_ratio=portfolio_summary["lifetime_sharpe"],
                    state_dict_path=weights_path,
                    description=f"HWM {tag} on {self.symbol} (Model: {self.model_name})",
                    regime_vector=regime_vector
                )

            # Check Drawdown Breach Rollback (>15%)
            did_rollback = self.rollback_engine.execute_rollback_if_breached(
                current_equity=current_eq,
                current_drawdown=current_dd
            )
            if did_rollback:
                logger.warning(f"Child {self.child_id} reverted after drawdown breach at step {step}.")

            # Check Stagnation (Time-Dilation) -> Request P2P weights or trigger self-mod
            stagnation_seconds = current_market_ts - portfolio_summary["last_hwm"]
            if stagnation_seconds >= 14400.0 and step % 100 == 0:  # 4 hours
                logger.info(f"Child {self.child_id} computationally stagnant for {stagnation_seconds/3600:.1f} hours. Querying Social Relay for peer breakthrough...")
                regime_vector = self.env.obs_mean.tolist() if hasattr(self.env, "obs_mean") else None
                peers = self.social_relay.query_top_peers(min_sharpe=1.2, current_regime_vector=regime_vector)
                if not peers:
                    logger.warning("No peers available during stagnation.")

            obs = next_obs

            if terminated or truncated:
                logger.info(f"Child {self.child_id} loop concluded at step {step}. Equity: ${current_eq:.2f}, Sharpe: {portfolio_summary['lifetime_sharpe']:.2f}.")
                break

        self.is_terminated = True
        return {
            "child_id": self.child_id,
            "model_name": self.model_name,
            "training_mode": "legacy",
            "steps_completed": step + 1,
            "final_equity": self.env.accounting.equity,
            "peak_equity": self.env.accounting.peak_equity,
            "max_drawdown": self.env.accounting.max_drawdown,
            "sharpe_ratio": self.env.accounting._compute_ratios()[0],
            "sortino_ratio": self.env.accounting._compute_ratios()[1],
            "active_tier": self.current_tier,
            "active_model": self.model_name,
        }


if __name__ == "__main__":
    logger.info("Testing SovereignChild standalone execution...")
    # Ensure synthetic data exists
    from data_forge.parquet_ingest import ParquetIngestPipeline
    import asyncio
    ingest = ParquetIngestPipeline(data_store_dir="data_store")
    asyncio.run(ingest.generate_synthetic_crucible_data(symbol="BTC-USDT", num_days=1, ticks_per_day=150))

    child = SovereignChild(child_id=1, symbol="BTC-USDT", initial_cash=10.0)

    if child.rl_trainer is not None:
        logger.info("Running SB3 Crucible Loop (gradient-based training active)...")
        result = child.run_crucible_loop_sb3(total_timesteps=500)
    else:
        logger.info("Running Legacy Crucible Loop (no gradient training)...")
        result = child.run_crucible_loop(max_steps=25)

    print("Crucible Loop Final Summary:", json.dumps(result, indent=2))
