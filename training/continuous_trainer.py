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
from scripts.deploy_config import DEPLOY_CONFIG


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
        bridge_live_data_every_cycles: int = 1,
    ):
        self.training_model_name = training_model_name
        self.data_store_dir = os.path.abspath(data_store_dir)
        self.state_dir = os.path.abspath(state_dir)
        self.deployed_model_path = deployed_model_path
        self.symbol = symbol
        self.train_interval = train_interval_minutes * 60  # Convert to seconds
        self.timesteps_per_cycle = timesteps_per_cycle
        self.inference_server = inference_server
        # PHASE 1 FIX: `DEPLOY_CONFIG.auto_promotion` used to be asserted at
        # construction and never actually checked -- _evaluate_and_promote()
        # promoted the instant the statistical gate passed, regardless of this
        # flag's value. See _evaluate_and_promote() and approve_pending_promotion()
        # below for the real gate. Keyed by agent_id; holds the in-memory champion
        # object (including its live SB3 model, needed for the EWC re-anchor step)
        # for any promotion awaiting human approval in THIS process.
        self.pending_promotions: Dict[int, Any] = {}

        # PHASE 2 FIX: "continuous" training used to mean continuous re-training on
        # the same synthetic-plus-stale-bulk data mix, cycle after cycle, regardless
        # of what actually happened in the live market during testnet/live operation
        # -- nothing bridged data_forge/lob_collector.py's live captures into the
        # data_store/processed/ tier TradeJackLOBEnv actually trains on. See
        # _bridge_live_data() below. bridge_live_data_every_cycles=1 means every
        # cycle by default; raise it to reduce redundant I/O on a fast train_interval
        # -- bridging is idempotent (safe to re-run on the same day) either way.
        self.bridge_live_data_every_cycles = max(1, bridge_live_data_every_cycles)

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

    def _bridge_live_data(self) -> None:
        """
        PHASE 2: closes the "real live market experience never reaches retraining"
        gap. Bridges today's and yesterday's data_forge/lob_collector.py live trade
        captures into TradeFlowPhysics's expected input location, then runs the
        existing, hardened batch bucketing pipeline over them -- see
        data_forge/live_trade_bridge.py's module docstring for why this reuses the
        batch pipeline rather than reimplementing an approximation of it.

        Both "today" and "yesterday" (UTC) are bridged every time this runs, not
        just whichever just turned over: "today" is normally still accumulating
        (only whatever hours have flushed so far), and re-bridging "yesterday" is
        what picks up its last few hours once the day is actually complete. Both
        operations are idempotent (bridge_live_trades_to_raw() always
        overwrites-then-atomic-renames the same output path for a given date), so
        there's no harm in repeating a day that hasn't changed.

        Deliberately does not know or care whether TradeJackLOBEnv re-globs
        data_store/processed/ per-episode or only at environment construction --
        that's physics/lob_env.py's concern, not this method's. This just
        guarantees the freshest possible physics file is on disk before this
        cycle's agent.train() calls run, which is the most this layer can
        responsibly promise without inspecting (or coupling to) the env's own
        internal caching behavior.

        Never allowed to crash the training loop: any failure here (a corrupt
        capture file, a permissions issue, polars/config unavailable, anything)
        is logged and skipped, not raised -- a bad live-data refresh should cost
        this cycle's freshness, not the whole continuous-training process.
        """
        try:
            from data_forge.live_trade_bridge import bridge_live_trades_to_raw
            from data_forge.feature_engineering import TradeFlowPhysics

            physics_engine = TradeFlowPhysics(self.symbol)
            now = time.gmtime()
            today = time.strftime("%Y-%m-%d", now)
            yesterday = time.strftime("%Y-%m-%d", time.gmtime(time.time() - 86400))

            for date in (yesterday, today):
                bridged = bridge_live_trades_to_raw(self.symbol, date, data_store_dir=self.data_store_dir)
                if bridged:
                    physics_engine.process_daily_file(date)
                    logger.info(f"Live data bridged and processed for {self.symbol} on {date}.")
        except Exception as e:
            logger.error(f"Live data bridge failed this cycle (training continues on existing data): {e}")

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

                if self.cycle_count % self.bridge_live_data_every_cycles == 0:
                    self._bridge_live_data()

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
                if getattr(DEPLOY_CONFIG, "auto_promotion", False):
                    self._promote_champion(champion)
                else:
                    self._queue_pending_promotion(champion, report)
            else:
                logger.info(f"Champion did not pass promotion gates: {report['verdict']}")

            eval_env.close()

        except Exception as e:
            logger.error(f"Evaluation error: {e}")

    def _pending_promotion_record_path(self, agent_id: int) -> str:
        return os.path.join(self.state_dir, "pending_promotions", f"agent_{agent_id}.json")

    def _queue_pending_promotion(self, champion, report: Dict[str, Any]) -> None:
        """
        PHASE 1 FIX: with `DEPLOY_CONFIG.auto_promotion=False` (the default), a
        champion that passes the statistical gate is no longer promoted immediately.
        It's queued here instead: recorded in-memory (self.pending_promotions, so
        approve_pending_promotion() can complete the full promotion including the
        EWC re-anchor step, which needs the live model object) and persisted to disk
        as a JSON record (so a human-facing tool -- dashboard, CLI, whatever -- has
        something durable to read and act on across process restarts, even though
        cross-process approval can only redo the checkpoint-copy step, not the EWC
        re-anchor -- see approve_pending_promotion()'s docstring for that limitation).
        """
        import json
        from datetime import datetime, timezone

        self.pending_promotions[champion.agent_id] = champion

        record_path = self._pending_promotion_record_path(champion.agent_id)
        os.makedirs(os.path.dirname(record_path), exist_ok=True)
        src = champion.checkpoint_path + ".zip"
        if not os.path.exists(src):
            src = champion.checkpoint_path
        record = {
            "agent_id": champion.agent_id,
            "label": champion.label,
            "checkpoint_path": src,
            "verdict": report.get("verdict"),
            "avg_sharpe": report.get("avg_sharpe"),
            "max_drawdown": report.get("max_drawdown"),
            "queued_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        with open(record_path, "w") as f:
            json.dump(record, f, indent=2)

        logger.warning(
            f"Champion Agent {champion.agent_id} ({champion.label}) PASSED promotion gates "
            f"but auto_promotion=False -- NOT promoted automatically. Pending record written "
            f"to '{record_path}'. Call trainer.approve_pending_promotion({champion.agent_id}) "
            f"(same process) or otherwise act on that record to actually deploy it."
        )

    def approve_pending_promotion(self, agent_id: int) -> bool:
        """
        Completes a promotion that was queued by _queue_pending_promotion(). This is
        the human-in-the-loop step `auto_promotion=False` is meant to require.

        Returns True if a promotion was completed, False if there was nothing pending
        for this agent_id (in-memory or on disk).
        """
        champion = self.pending_promotions.pop(agent_id, None)
        record_path = self._pending_promotion_record_path(agent_id)

        if champion is not None:
            # Full path: same process the evaluation ran in, so champion.trainer.model
            # is still a live object -- _promote_champion can do the complete job,
            # EWC re-anchor included.
            self._promote_champion(champion)
            if os.path.exists(record_path):
                os.remove(record_path)
            return True

        if os.path.exists(record_path):
            # Cross-process approval: the champion object (and its live model) no
            # longer exists in this process, so this can only redo the checkpoint
            # copy + hot-swap, not the EWC re-anchor step. Stated plainly rather than
            # silently skipped, since it's a real (if minor) difference in behavior
            # from same-process approval.
            import json

            with open(record_path, "r") as f:
                record = json.load(f)
            src = record["checkpoint_path"]
            if not os.path.exists(src):
                logger.error(f"Pending promotion checkpoint not found: {src}")
                return False
            dst = self.deployed_model_path + ".zip"
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)
            self.promotion_count += 1
            logger.warning(
                f"Approved cross-process promotion for agent {agent_id}: {src} -> {dst}. "
                f"NOTE: EWC re-anchor was skipped (no live model object in this process) -- "
                f"the next training cycle will regularize against whatever was last "
                f"in-process-promoted, not this checkpoint, until that happens naturally."
            )
            if self.inference_server and hasattr(self.inference_server, "hot_swap_model"):
                self.inference_server.hot_swap_model(self.deployed_model_path)
            os.remove(record_path)
            return True

        logger.warning(f"No pending promotion found for agent_id={agent_id}.")
        return False

    def list_pending_promotions(self) -> Dict[int, Dict[str, Any]]:
        """Reads every on-disk pending-promotion record, for a dashboard/CLI to show
        an operator what's awaiting approval, without needing this process's memory."""
        import json

        out = {}
        pending_dir = os.path.join(self.state_dir, "pending_promotions")
        if not os.path.isdir(pending_dir):
            return out
        for fname in os.listdir(pending_dir):
            if fname.endswith(".json"):
                with open(os.path.join(pending_dir, fname)) as f:
                    record = json.load(f)
                out[record["agent_id"]] = record
        return out

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
