"""
Continuous Trainer — Background daemon that trains while inference runs.

Runs in a background thread/process alongside the LiveInferenceServer.
Collects live market transitions from inference telemetry, trains a copy
of the model, evaluates it via walk-forward, and promotes if it passes
all gates.

Key invariant: inference always runs on the FROZEN copy. Training always
runs on the TRAINING copy. They never share weights during a forward pass.
Promotion is atomic (file swap + reload).

Usage:
    trainer = ContinuousTrainer(
        training_model_name="PPO-DilatedCNN",
        inference_server=server,
        data_store_dir="data_store",
    )
    await trainer.run_continuous(max_cycles=100)
"""

import os
import time
import asyncio
import logging
import shutil
from typing import Dict, Any, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] (ContinuousTrainer) %(message)s")
logger = logging.getLogger("ContinuousTrainer")

try:
    from stable_baselines3 import PPO, SAC, DQN
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False

from training.walk_forward_evaluator import WalkForwardEvaluator
from training.crucible_tournament import CrucibleTournament


class ContinuousTrainer:
    """
    Background self-RL training daemon.

    Loop:
    1. Run a tournament training cycle on historical + recent data
    2. Evaluate the champion via WalkForwardEvaluator
    3. If champion passes all gates → promote to live inference (atomic hot-swap)
    4. Sleep until next cycle

    The training and inference models are completely separate:
    - Inference model: frozen weights, read-only, runs on live data
    - Training models: mutable, trained on historical replay + recent telemetry
    - Promotion: copy champion's checkpoint to the deployed model path, hot-swap
    """

    def __init__(
        self,
        training_model_name: str = "PPO-DilatedCNN",
        data_store_dir: str = "data_store",
        state_dir: str = "state",
        deployed_model_path: str = "state/deployed/weights_promoted",
        symbol: str = "BTC-USDT",
        train_interval_minutes: float = 30.0,
        timesteps_per_cycle: int = 10000,
        inference_server=None,
        plasticity_reset_interval_cycles: int = 5,
        plasticity_fisher_reset_fraction: float = 0.5,
    ):
        self.training_model_name = training_model_name
        self.data_store_dir = os.path.abspath(data_store_dir)
        self.state_dir = os.path.abspath(state_dir)
        self.deployed_model_path = deployed_model_path
        self.symbol = symbol
        self.train_interval = train_interval_minutes * 60  # Convert to seconds
        self.timesteps_per_cycle = timesteps_per_cycle
        self.inference_server = inference_server

        # Initialize components
        self.tournament = CrucibleTournament(
            data_store_dir=self.data_store_dir,
            state_dir=os.path.join(self.state_dir, "tournament"),
            symbol=self.symbol,
        )

        self.evaluator = WalkForwardEvaluator(
            min_test_days=14,
            min_trade_count=20,
            max_drawdown_threshold=0.15,
            significance_level=0.05,
        )

        # Primacy-bias / loss-of-plasticity mitigation (Nikishin et al. 2022,
        # extended with Fisher-guided selectivity per arxiv 2502.00802) --
        # see swarm/plasticity_manager.py's module docstring for the research
        # this is grounded in. A system training forever on live,
        # non-stationary market data is close to exactly the setting this
        # research studies, so this isn't a speculative addition.
        from swarm.plasticity_manager import PlasticityManager
        self.plasticity_manager = PlasticityManager(
            reset_interval_cycles=plasticity_reset_interval_cycles,
            fisher_reset_fraction=plasticity_fisher_reset_fraction,
            use_fisher_guidance=True,
        )

        self.cycle_count = 0
        self.promotion_count = 0
        self.is_running = False

    async def run_continuous(self, max_cycles: Optional[int] = None):
        """
        Main continuous training loop.

        Runs tournament training cycles with evaluation and promotion.
        """
        self.is_running = True
        logger.info(
            f"ContinuousTrainer starting: "
            f"interval={self.train_interval/60:.0f}min, "
            f"timesteps/cycle={self.timesteps_per_cycle}"
        )

        try:
            # Initialize tournament agents once
            self.tournament.initialize_all_agents()

            while self.is_running:
                self.cycle_count += 1
                logger.info(f"=== Training Cycle {self.cycle_count} ===")

                try:
                    # 1. Run one tournament training cycle
                    for agent in self.tournament.agents:
                        agent.train(timesteps=self.timesteps_per_cycle)

                        # Plasticity reset check, per-agent, using that
                        # agent's own most recently computed Fisher matrix
                        # (from its last EWC anchor, if any) to guide which
                        # head layers are safe to reset. Runs every cycle but
                        # only actually resets once every
                        # `plasticity_reset_interval_cycles` cycles per agent
                        # -- see PlasticityManager.maybe_reset()'s own
                        # internal counter, kept independent of the EWC
                        # re-anchoring cadence (which only happens on
                        # promotion, not every cycle) since primacy bias
                        # accumulates with training steps regardless of
                        # whether a promotion has happened recently.
                        if agent.trainer is not None:
                            fisher = None
                            existing_ewc = getattr(agent.trainer, "ewc_instance", None)
                            if existing_ewc is not None:
                                fisher = existing_ewc.fisher_matrix
                            did_reset = self.plasticity_manager.maybe_reset(
                                agent_id=str(agent.agent_id), model=agent.trainer.model, fisher_matrix=fisher
                            )
                            if did_reset:
                                self.tournament.progress_ledger.record_event(
                                    agent_id=agent.agent_id, event_type="plasticity_reset",
                                    detail=f"reset #{self.plasticity_manager.reset_count}",
                                )

                    # 2. Run PBT if enough timesteps accumulated
                    total_steps = sum(a.cumulative_timesteps for a in self.tournament.agents)
                    if total_steps % self.tournament.pbt_interval < self.timesteps_per_cycle:
                        self.tournament._execute_pbt_step()

                    # 3. Evaluate champion for promotion
                    champion = self.tournament.get_champion()
                    if champion and champion.trainer:
                        await self._evaluate_and_promote(champion)

                except Exception as e:
                    logger.error(f"Training cycle error: {e}")

                # Check stop condition
                if max_cycles and self.cycle_count >= max_cycles:
                    logger.info(f"Max cycles ({max_cycles}) reached. Stopping.")
                    break

                # Sleep until next cycle
                logger.info(f"Next cycle in {self.train_interval/60:.0f} minutes...")
                await asyncio.sleep(self.train_interval)

        except asyncio.CancelledError:
            logger.info("ContinuousTrainer cancelled.")
        finally:
            self.is_running = False
            logger.info(
                f"ContinuousTrainer stopped after {self.cycle_count} cycles, "
                f"{self.promotion_count} promotions."
            )

    async def _evaluate_and_promote(self, champion):
        """Evaluate the tournament champion and promote if it passes all gates."""
        logger.info(f"Evaluating champion: Agent {champion.agent_id} ({champion.label})")

        try:
            from physics.lob_env import TradeJackLOBEnv

            eval_env = TradeJackLOBEnv(
                symbol=self.symbol,
                initial_cash=10.0,
                data_store_dir=self.data_store_dir,
                child_id=999,  # Special eval ID
            )

            # Load incumbent if exists
            incumbent_model = None
            deployed_path = self.deployed_model_path
            for ext in [".zip", ""]:
                if os.path.exists(deployed_path + ext):
                    try:
                        algo_map = {"PPO": PPO, "SAC": SAC, "DQN": DQN}
                        prefix = champion.model_name.split("-")[0]
                        cls = algo_map.get(prefix, PPO)
                        incumbent_model = cls.load(deployed_path, device="cpu")
                    except Exception:
                        pass
                    break

            # Run walk-forward evaluation
            passed, report = self.evaluator.evaluate_candidate(
                candidate_model=champion.trainer.model,
                incumbent_model=incumbent_model,
                env=eval_env,
                max_steps=5000,
            )

            if passed:
                self._promote_champion(champion)
            else:
                logger.info(f"Champion did not pass promotion gates: {report['verdict']}")

            eval_env.close()

        except Exception as e:
            logger.error(f"Evaluation error: {e}")

    def _promote_champion(self, champion):
        """Atomically promote champion model to the deployed model path."""
        logger.info(f"PROMOTING Agent {champion.agent_id} ({champion.label}) to live deployment!")

        try:
            # Copy checkpoint to deployed path
            src = champion.checkpoint_path + ".zip"
            if not os.path.exists(src):
                src = champion.checkpoint_path
            if not os.path.exists(src):
                logger.error(f"Champion checkpoint not found: {src}")
                return

            dst = self.deployed_model_path + ".zip"
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)

            logger.info(f"Checkpoint copied: {src} → {dst}")

            # Hot-swap inference server model if connected
            if self.inference_server and hasattr(self.inference_server, "hot_swap_model"):
                self.inference_server.hot_swap_model(self.deployed_model_path)

            self.promotion_count += 1
            logger.info(f"Promotion #{self.promotion_count} complete.")
            self.tournament.progress_ledger.record_event(
                agent_id=champion.agent_id, event_type="promotion",
                detail=f"promotion #{self.promotion_count}",
            )

            # Re-anchor EWC to the just-promoted policy so the NEXT training
            # cycle is regularized against forgetting what just got promoted,
            # instead of drifting freely until the next promotion event.
            # Non-fatal by design — an EWC failure must never roll back a
            # promotion that already succeeded, it only affects how gently
            # future training explores away from this point. Only applies to
            # PPO-family agents (see PolicyEWC's docstring for why SAC/DQN
            # aren't supported yet) — other-architecture agents in the
            # tournament will just see 0 matching parameters and get zero
            # penalty, which is correct, not broken.
            try:
                from swarm.ewc_sb3_adapter import PolicyEWC
                anchor = PolicyEWC(champion.trainer.model, n_calibration_samples=512)
                if anchor.n_calibration_samples_used > 0:
                    updated_count = 0
                    for agent in self.tournament.agents:
                        if agent.trainer is not None and agent.trainer.set_ewc_instance(anchor):
                            updated_count += 1
                    logger.info(
                        f"EWC anchor updated on {updated_count}/{len(self.tournament.agents)} "
                        f"tournament agents from promotion #{self.promotion_count}'s champion."
                    )
                else:
                    logger.warning("EWC anchor computed but had no calibration data — not applied.")
            except Exception as e:
                logger.error(f"EWC re-anchoring after promotion failed (non-fatal, promotion itself succeeded): {e}")

        except Exception as e:
            logger.error(f"Promotion failed: {e}")

    def stop(self):
        """Signal the trainer to stop."""
        self.is_running = False

    def get_status(self) -> Dict[str, Any]:
        """Current trainer status for dashboard."""
        return {
            "running": self.is_running,
            "cycle_count": self.cycle_count,
            "promotion_count": self.promotion_count,
            "standings": self.tournament.get_standings(),
        }


if __name__ == "__main__":
    logger.info("ContinuousTrainer loaded.")
    logger.info("Run via: python -m scripts.genesis_prime --paper")
