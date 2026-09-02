"""
Online RL Trainer wrapping stable-baselines3 PPO/SAC/DQN.

This is the training engine that didn't exist in TradeJack v1-v2.
It provides actual gradient-based learning using validated SB3 implementations,
while keeping our custom code limited to:
  - LOB observation encoding (shared_encoder.py)
  - HER re-labeling (replay_buffer.py)
  - EWC penalty injection (ewc_optimizer.py via SB3 callback)
  - PBT checkpoint management (rl_mechanics.py)

The trainer operates in two modes:
1. Crucible mode: SB3 owns the env loop via model.learn()
2. Inference mode: model.predict() only, frozen weights

Usage:
    trainer = OnlineRLTrainer(algo="ppo", env=lob_env)
    trainer.learn(total_timesteps=10000)
    action = trainer.predict(obs)
    trainer.save("checkpoints/model_v1")
"""

import os
import time
import logging
import numpy as np
from typing import Dict, Any, Optional, List, Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] (RLTrainer) %(message)s")
logger = logging.getLogger("RLTrainer")

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from stable_baselines3 import PPO, SAC, DQN
    from stable_baselines3.common.callbacks import BaseCallback, CallbackList
    from stable_baselines3.common.logger import configure as configure_sb3_logger
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False

from swarm.model_registry import REGISTRY, ModelCard


class TrainingMetricsCallback(BaseCallback):
    """
    SB3 callback that collects training metrics for monitoring.
    Logs loss, gradient norms, entropy, and learning rate at each training step.
    """

    def __init__(self, log_interval: int = 100, verbose: int = 0):
        super().__init__(verbose)
        self.log_interval = log_interval
        self.metrics_history: List[Dict[str, float]] = []
        self._step_count = 0

    def _on_step(self) -> bool:
        self._step_count += 1
        return True

    def _on_rollout_end(self) -> None:
        """Collect metrics after each rollout (PPO) or training step (SAC/DQN)."""
        metrics = {"timestep": self.num_timesteps, "wall_time": time.time()}

        # Extract loss and other metrics from SB3's internal logger
        if hasattr(self.model, "logger") and self.model.logger is not None:
            try:
                name_to_value = getattr(self.model.logger, "name_to_value", {})
                for key in ["train/loss", "train/policy_gradient_loss",
                            "train/value_loss", "train/entropy_loss",
                            "train/approx_kl", "train/clip_fraction",
                            "train/learning_rate"]:
                    if key in name_to_value:
                        metrics[key.replace("train/", "")] = name_to_value[key]
            except Exception:
                pass

        # Compute gradient norm across policy parameters
        if TORCH_AVAILABLE and hasattr(self.model, "policy"):
            try:
                total_norm = 0.0
                param_count = 0
                for p in self.model.policy.parameters():
                    if p.grad is not None:
                        total_norm += p.grad.data.norm(2).item() ** 2
                        param_count += 1
                if param_count > 0:
                    metrics["grad_norm"] = total_norm ** 0.5
            except Exception:
                pass

        self.metrics_history.append(metrics)

        if self._step_count % self.log_interval == 0 and self.verbose > 0:
            loss = metrics.get("loss", metrics.get("policy_gradient_loss", "N/A"))
            grad = metrics.get("grad_norm", "N/A")
            logger.info(
                f"[Train] step={self.num_timesteps} "
                f"loss={loss} grad_norm={grad}"
            )

    def get_recent_metrics(self, n: int = 10) -> List[Dict[str, float]]:
        return self.metrics_history[-n:]


class OnlineRLTrainer:
    """
    Wraps stable-baselines3 for online RL training inside the TradeJack Crucible.

    Responsibilities (ours):
      - LOB observation preprocessing (shared_encoder.py via SB3 policy_kwargs)
      - EWC penalty injection via callback
      - Checkpoint save/load for PBT weight inheritance
      - Training metrics collection

    Responsibilities (SB3):
      - PPO clip loss, GAE advantage estimation
      - SAC entropy-regularized policy optimization
      - DQN TD-error with prioritized replay
      - Optimizer state, learning rate schedules
    """

    def __init__(
        self,
        model_name: str = "PPO-DilatedCNN",
        env=None,
        ewc_instance=None,
        device: str = "auto",
        log_dir: Optional[str] = None,
        **override_hyperparams,
    ):
        if not SB3_AVAILABLE:
            raise ImportError(
                "stable-baselines3 required for OnlineRLTrainer. "
                "pip install stable-baselines3"
            )
        if env is None:
            raise ValueError("OnlineRLTrainer requires a Gymnasium-compliant env.")

        self.model_name = model_name
        self.env = env
        self.ewc_instance = ewc_instance
        self.device = device

        # Build the SB3 model via registry
        self.model = REGISTRY.build_model(
            model_name=model_name,
            env=env,
            ewc_instance=ewc_instance,
            device=device,
            **override_hyperparams,
        )

        # Set up training metrics callback
        self.metrics_callback = TrainingMetricsCallback(log_interval=50, verbose=1)

        # Aggregate all callbacks (EWC + metrics)
        all_callbacks = [self.metrics_callback]
        if hasattr(self.model, "_tradejack_callbacks"):
            all_callbacks.extend(self.model._tradejack_callbacks)
        self.callback_list = CallbackList(all_callbacks)

        # Configure SB3 logging
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            new_logger = configure_sb3_logger(log_dir, ["csv", "tensorboard"])
            self.model.set_logger(new_logger)

        self.total_timesteps_trained = 0
        self.card = REGISTRY.get_model_card(model_name)

        logger.info(
            f"OnlineRLTrainer initialized: model={model_name}, "
            f"algo={self.card.algo_class if self.card else 'unknown'}, "
            f"device={device}"
        )

    def learn(self, total_timesteps: int = 10000, reset_num_timesteps: bool = False) -> Dict[str, Any]:
        """
        Run SB3's training loop for the given number of timesteps.

        SB3 owns the environment interaction — it collects rollouts and trains.
        This is the main training entry point for the Crucible.

        Returns:
            dict with training summary metrics
        """
        start_time = time.time()
        start_metrics_count = len(self.metrics_callback.metrics_history)

        logger.info(f"Starting training: {total_timesteps} timesteps...")

        self.model.learn(
            total_timesteps=total_timesteps,
            callback=self.callback_list,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=False,
        )

        self.total_timesteps_trained += total_timesteps
        elapsed = time.time() - start_time

        # Collect summary
        new_metrics = self.metrics_callback.metrics_history[start_metrics_count:]
        summary = {
            "timesteps_trained": total_timesteps,
            "total_timesteps": self.total_timesteps_trained,
            "wall_time_sec": round(elapsed, 2),
            "steps_per_sec": round(total_timesteps / max(elapsed, 1e-6), 1),
            "num_updates": len(new_metrics),
        }

        if new_metrics:
            # Average metrics across updates
            for key in ["loss", "policy_gradient_loss", "value_loss",
                        "entropy_loss", "grad_norm"]:
                values = [m[key] for m in new_metrics if key in m]
                if values:
                    summary[f"avg_{key}"] = round(np.mean(values), 6)

        logger.info(
            f"Training complete: {total_timesteps} steps in {elapsed:.1f}s "
            f"({summary['steps_per_sec']} steps/s)"
        )

        return summary

    def predict(self, obs: Dict[str, np.ndarray], deterministic: bool = True) -> float:
        """
        Run inference on a single observation.

        Returns target position fraction in [-1, 1].
        For DQN (discrete), maps discrete action index back to continuous.
        """
        action, _states = self.model.predict(obs, deterministic=deterministic)

        # DQN returns discrete index — map back to continuous
        if self.card and self.card.algo_class == "DQN":
            from swarm.model_registry import DiscreteActionWrapper
            action_val = DiscreteActionWrapper.DISCRETE_ACTIONS[int(action)]
            return float(action_val)

        # PPO/SAC return continuous action
        if isinstance(action, np.ndarray):
            return float(np.clip(action[0], -1.0, 1.0))
        return float(np.clip(action, -1.0, 1.0))

    def save(self, path: str):
        """Save model checkpoint (weights + optimizer state) for PBT inheritance."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        self.model.save(path)
        logger.info(f"Model saved to {path}")

    def load(self, path: str, env=None):
        """Load model checkpoint. Optionally attach to a new environment."""
        target_env = env or self.env

        algo_cls = {"PPO": PPO, "SAC": SAC, "DQN": DQN}
        cls = algo_cls.get(self.card.algo_class if self.card else "PPO", PPO)

        self.model = cls.load(path, env=target_env, device=self.device)
        logger.info(f"Model loaded from {path}")

    def get_policy_parameters(self):
        """Get the policy network's parameters (for EWC Fisher computation)."""
        if hasattr(self.model, "policy"):
            return self.model.policy.parameters()
        return iter([])

    def get_training_metrics(self, n: int = 10) -> List[Dict[str, float]]:
        """Get the most recent N training metric snapshots."""
        return self.metrics_callback.get_recent_metrics(n)

    def get_model_state_summary(self) -> Dict[str, Any]:
        """Summary of current model state for PBT and monitoring."""
        summary = {
            "model_name": self.model_name,
            "algo_class": self.card.algo_class if self.card else "unknown",
            "total_timesteps": self.total_timesteps_trained,
            "device": str(self.device),
        }

        # Count parameters
        if TORCH_AVAILABLE and hasattr(self.model, "policy"):
            total_params = sum(p.numel() for p in self.model.policy.parameters())
            trainable_params = sum(p.numel() for p in self.model.policy.parameters() if p.requires_grad)
            summary["total_params"] = total_params
            summary["trainable_params"] = trainable_params

        return summary


if __name__ == "__main__":
    logger.info("OnlineRLTrainer requires a Gymnasium env to test.")
    logger.info("Run via: python -m scripts.genesis_prime --agents=1 --max-steps=100")
    logger.info("Or test with: python -m pytest tests/test_rl_trainer.py -v")

    if SB3_AVAILABLE:
        logger.info(f"SB3 version: {PPO.__module__}")
        logger.info("SB3 available — trainer ready.")
    else:
        logger.error("SB3 not installed. Run: pip install stable-baselines3")
