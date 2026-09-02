"""
Crucible Tournament — Multi-agent training competition with PBT.

Runs 3-5 agents in parallel, each with different model architectures/hyperparameters,
competing on the same historical + recent live data. PBT operates across agents
every 6 hours, with bottom performers inheriting top performers' weights.

Winners are submitted to the WalkForwardEvaluator for live promotion candidacy.

Architecture ablations are exposed as PBT hyperparameters (not separate classes),
keeping the tournament focused on finding what works rather than maintaining
a zoo of models.

Usage:
    tournament = CrucibleTournament(data_store_dir="data_store")
    results = tournament.run_tournament(total_timesteps=100000)
    champion = tournament.get_champion()
"""

import os
import time
import logging
import json
import shutil
from typing import Dict, Any, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] (Tournament) %(message)s")
logger = logging.getLogger("Tournament")

try:
    from stable_baselines3 import PPO, SAC, DQN
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False

from swarm.model_registry import REGISTRY
from swarm.rl_mechanics import PopulationBasedTrainingEngine


# Default tournament roster
DEFAULT_ROSTER = [
    {"model_name": "PPO-DilatedCNN", "label": "PPO-CNN"},
    {"model_name": "SAC-DilatedCNN", "label": "SAC-CNN"},
    {"model_name": "DuelingDQN",     "label": "DQN"},
    {"model_name": "PPO-DilatedCNN", "label": "PPO-CNN-v2", "learning_rate": 1e-4},  # Hyperparameter ablation
]


class TournamentAgent:
    """A single agent in the tournament with its own model, env, and training state."""

    def __init__(
        self,
        agent_id: int,
        model_name: str,
        label: str,
        data_store_dir: str,
        state_dir: str,
        **override_hyperparams,
    ):
        self.agent_id = agent_id
        self.model_name = model_name
        self.label = label
        self.state_dir = state_dir
        self.override_hyperparams = override_hyperparams

        self.checkpoint_dir = os.path.join(state_dir, f"agent_{agent_id}")
        self.checkpoint_path = os.path.join(self.checkpoint_dir, "model")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # These are set during training
        self.trainer = None
        self.env = None
        self.last_train_result: Dict[str, Any] = {}
        self.cumulative_timesteps = 0
        self.best_sortino = -999.0

    def initialize(self, data_store_dir: str, symbol: str = "BTC-USDT"):
        """Create the env and trainer for this agent."""
        from physics.lob_env import TradeJackLOBEnv
        from swarm.rl_trainer import OnlineRLTrainer

        self.env = TradeJackLOBEnv(
            symbol=symbol,
            initial_cash=10.0,
            data_store_dir=data_store_dir,
            child_id=self.agent_id,
        )

        log_dir = os.path.join(self.state_dir, f"agent_{self.agent_id}", "tb_logs")
        self.trainer = OnlineRLTrainer(
            model_name=self.model_name,
            env=self.env,
            device="auto",
            log_dir=log_dir,
            **self.override_hyperparams,
        )

    def train(self, timesteps: int = 10000) -> Dict[str, Any]:
        """Run one training cycle."""
        if self.trainer is None:
            logger.error(f"Agent {self.agent_id} not initialized.")
            return {"error": "not initialized"}

        try:
            result = self.trainer.learn(total_timesteps=timesteps)
            self.cumulative_timesteps += timesteps
            self.last_train_result = result

            # Save checkpoint
            self.trainer.save(self.checkpoint_path)

            # Get performance metrics from env
            sortino = 0.0
            if hasattr(self.env, "accounting"):
                _, sortino = self.env.accounting._compute_ratios()
                self.best_sortino = max(self.best_sortino, sortino)

            result.update({
                "agent_id": self.agent_id,
                "label": self.label,
                "model_name": self.model_name,
                "cumulative_timesteps": self.cumulative_timesteps,
                "sortino_ratio": sortino,
                "best_sortino": self.best_sortino,
                "equity": self.env.accounting.equity if hasattr(self.env, "accounting") else 0.0,
            })

            logger.info(
                f"[Agent {self.agent_id} / {self.label}] "
                f"Trained {timesteps} steps. Sortino={sortino:.3f}, "
                f"Equity=${result.get('equity', 0):.2f}"
            )

            return result

        except Exception as e:
            logger.error(f"Agent {self.agent_id} training error: {e}")
            return {"agent_id": self.agent_id, "error": str(e)}

    def get_status(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "label": self.label,
            "model_name": self.model_name,
            "cumulative_timesteps": self.cumulative_timesteps,
            "best_sortino": self.best_sortino,
            "checkpoint_path": self.checkpoint_path,
        }


class CrucibleTournament:
    """
    Multi-agent training tournament with Population-Based Training.

    Runs N agents in parallel, each training on the same market data but with
    different architectures/hyperparameters. PBT operates every `pbt_interval`
    timesteps, with bottom performers inheriting top performers' checkpoints.
    """

    def __init__(
        self,
        data_store_dir: str = "d:/TradeJack/data_store",
        state_dir: str = "d:/TradeJack/state/tournament",
        symbol: str = "BTC-USDT",
        roster: Optional[List[Dict[str, Any]]] = None,
        pbt_interval: int = 50000,
        pbt_exploit_fraction: float = 0.3,
    ):
        self.data_store_dir = os.path.abspath(data_store_dir)
        self.state_dir = os.path.abspath(state_dir)
        self.symbol = symbol
        self.roster_config = roster or DEFAULT_ROSTER
        self.pbt_interval = pbt_interval
        self.pbt_engine = PopulationBasedTrainingEngine(
            swarm_size=len(self.roster_config),
            exploit_fraction=pbt_exploit_fraction,
        )

        os.makedirs(self.state_dir, exist_ok=True)

        # Initialize agents
        self.agents: List[TournamentAgent] = []
        for i, cfg in enumerate(self.roster_config):
            hp = {k: v for k, v in cfg.items() if k not in ("model_name", "label")}
            agent = TournamentAgent(
                agent_id=i,
                model_name=cfg["model_name"],
                label=cfg.get("label", f"agent_{i}"),
                data_store_dir=self.data_store_dir,
                state_dir=self.state_dir,
                **hp,
            )
            self.agents.append(agent)

        self.champion_id: Optional[int] = None
        self.history: List[Dict[str, Any]] = []

    def initialize_all_agents(self):
        """Create envs and trainers for all agents."""
        logger.info(f"Initializing {len(self.agents)} tournament agents...")
        for agent in self.agents:
            agent.initialize(self.data_store_dir, self.symbol)
        logger.info("All agents initialized.")

    def run_tournament(
        self,
        total_timesteps: int = 100000,
        cycle_timesteps: int = 10000,
    ) -> Dict[str, Any]:
        """
        Run the full tournament.

        Training proceeds in cycles of `cycle_timesteps`. PBT runs every
        `pbt_interval` total timesteps.
        """
        self.initialize_all_agents()

        logger.info(
            f"Tournament starting: {len(self.agents)} agents, "
            f"{total_timesteps} total timesteps, "
            f"PBT every {self.pbt_interval} steps"
        )

        total_trained = 0
        next_pbt_at = self.pbt_interval
        cycle_num = 0

        while total_trained < total_timesteps:
            cycle_num += 1
            remaining = total_timesteps - total_trained
            steps_this_cycle = min(cycle_timesteps, remaining)

            logger.info(f"--- Tournament Cycle {cycle_num} ({total_trained}/{total_timesteps} steps) ---")

            # Train all agents sequentially (parallel with ThreadPoolExecutor if CPU-bound)
            cycle_results = []
            for agent in self.agents:
                result = agent.train(timesteps=steps_this_cycle)
                cycle_results.append(result)

            total_trained += steps_this_cycle

            # Run PBT if interval reached
            if total_trained >= next_pbt_at:
                self._execute_pbt_step()
                next_pbt_at += self.pbt_interval

            # Record history
            self.history.append({
                "cycle": cycle_num,
                "total_trained": total_trained,
                "results": cycle_results,
            })

        # Determine champion
        best = max(self.agents, key=lambda a: a.best_sortino)
        self.champion_id = best.agent_id

        summary = {
            "total_cycles": cycle_num,
            "total_timesteps": total_trained,
            "champion": best.get_status(),
            "standings": [a.get_status() for a in self.agents],
        }

        logger.info(f"Tournament complete. Champion: Agent {best.agent_id} ({best.label}) "
                     f"with Sortino={best.best_sortino:.3f}")

        # Save tournament results
        results_path = os.path.join(self.state_dir, "tournament_results.json")
        with open(results_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)

        return summary

    def _execute_pbt_step(self):
        """Run PBT across tournament agents."""
        logger.info("Executing PBT step across tournament population...")

        population_status = []
        for agent in self.agents:
            population_status.append({
                "child_id": agent.agent_id,
                "model_name": agent.model_name,
                "equity": agent.last_train_result.get("equity", 10.0),
                "sortino_ratio": agent.last_train_result.get("sortino_ratio", 0.0),
                "ticks_active": agent.cumulative_timesteps,
                "learning_rate": agent.override_hyperparams.get("learning_rate", 3e-4),
                "checkpoint_path": agent.checkpoint_path,
            })

        sorted_pop = self.pbt_engine.execute_pbt_step(population_status)

        # Apply PBT mutations: copy checkpoints, update hyperparams
        for updated in sorted_pop:
            agent = self.agents[updated["child_id"]]
            if "parent_lineage" in updated:
                parent_id = updated["parent_lineage"]
                parent_agent = self.agents[parent_id]

                # Copy parent checkpoint to child
                src = parent_agent.checkpoint_path + ".zip"
                dst = agent.checkpoint_path + ".zip"
                if os.path.exists(src) and src != dst:
                    try:
                        shutil.copy2(src, dst)
                        # Reload model from inherited checkpoint
                        if agent.trainer:
                            agent.trainer.load(agent.checkpoint_path, env=agent.env)
                        logger.info(f"PBT: Agent {agent.agent_id} inherited from Agent {parent_id}")
                    except Exception as e:
                        logger.warning(f"PBT checkpoint copy failed: {e}")

                # Update learning rate if mutated
                if "learning_rate" in updated:
                    agent.override_hyperparams["learning_rate"] = updated["learning_rate"]

    def get_champion(self) -> Optional[TournamentAgent]:
        """Get the tournament champion (best performing agent)."""
        if self.champion_id is not None:
            return self.agents[self.champion_id]
        if self.agents:
            return max(self.agents, key=lambda a: a.best_sortino)
        return None

    def get_standings(self) -> List[Dict[str, Any]]:
        """Current tournament standings."""
        standings = [a.get_status() for a in self.agents]
        standings.sort(key=lambda x: x["best_sortino"], reverse=True)
        return standings


if __name__ == "__main__":
    logger.info("CrucibleTournament loaded.")
    logger.info(f"Default roster: {len(DEFAULT_ROSTER)} agents")
    for cfg in DEFAULT_ROSTER:
        logger.info(f"  {cfg['label']}: {cfg['model_name']}")
