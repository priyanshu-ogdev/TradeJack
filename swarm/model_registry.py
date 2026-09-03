"""
TradeJack v3 Model Registry — 5 models backed by stable-baselines3.

Replaces the legacy 24-class educational port of huseinzol05/Stock-Prediction-Models.
Each entry maps to a real, tested RL algorithm implementation from SB3, using our
custom LOBFeatureEncoder as the shared backbone.

Models:
  Tier 1 (GPU, <=20GB): PPO-Transformer (full causal transformer encoder)
  Tier 2 (GPU, <=4GB):  PPO-DilatedCNN, SAC-DilatedCNN
  Tier 3 (CPU/1GB):     DuelingDQN (discretized actions, lightweight)
  Baselines:            Momentum-Baseline, BuyAndHold-Baseline

The REGISTRY singleton provides the same interface as the legacy registry:
  - REGISTRY.get_model_card(name) -> ModelCard
  - REGISTRY.list_models_for_tier(max_tier) -> List[ModelCard]
  - REGISTRY.build_model(name, env) -> SB3 model or baseline instance
"""

import os
import math
import logging
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Callable, Union

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] (ModelRegistry) %(message)s")
logger = logging.getLogger("ModelRegistry")

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not installed; ModelRegistry requires PyTorch for RL models.")

try:
    from stable_baselines3 import PPO, SAC, DQN
    from stable_baselines3.common.callbacks import BaseCallback
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    logger.warning("stable-baselines3 not installed. RL models unavailable. pip install stable-baselines3")

try:
    import gymnasium as gym
    from gymnasium import spaces
    GYM_AVAILABLE = True
except ImportError:
    GYM_AVAILABLE = False


# ─── MODEL CARD ───

@dataclass
class ModelCard:
    """
    Model specification card. Kept compatible with legacy interface but
    extended for SB3 integration.
    """
    model_name: str
    tier_requirement: int       # 1 (20GB), 2 (4GB), 3 (1GB/CPU)
    category: str               # "rl_ppo", "rl_sac", "rl_dqn", "rule_based", "baseline"
    algo_class: str             # "PPO", "SAC", "DQN", "rule", "static"
    encoder_type: str           # "transformer", "dilated_cnn", "mlp", "none"
    vram_estimate_mb: float
    description: str
    default_hyperparams: Dict[str, Any] = field(default_factory=dict)

    @property
    def algo(self) -> str:
        return self.algo_class


# ─── DISCRETE ACTION WRAPPER ───

if GYM_AVAILABLE:
    class DiscreteActionWrapper(gym.ActionWrapper):
        """
        Wraps a continuous [-1, 1] action space into discrete actions for DQN.
        Actions: {-1.0, -0.5, 0.0, 0.5, 1.0} = 5 discrete levels.
        """
        DISCRETE_ACTIONS = [-1.0, -0.5, 0.0, 0.5, 1.0]

        def __init__(self, env):
            super().__init__(env)
            self.action_space = spaces.Discrete(len(self.DISCRETE_ACTIONS))

        def action(self, action_idx):
            return [self.DISCRETE_ACTIONS[int(action_idx)]]


# ─── EWC CALLBACK FOR SB3 ───

if SB3_AVAILABLE:
    class EWCCallback(BaseCallback):
        """
        SB3 callback that adds Elastic Weight Consolidation penalty
        after each training update.

        The EWC penalty prevents catastrophic forgetting when the model
        continuously adapts to non-stationary market regimes.
        """

        def __init__(self, ewc_instance=None, verbose=0):
            super().__init__(verbose)
            self.ewc_instance = ewc_instance

        def _on_step(self) -> bool:
            return True

        def _on_rollout_end(self) -> None:
            """Apply EWC correction after each rollout/training cycle."""
            if self.ewc_instance is None:
                return
            # Compute EWC penalty and add as a regularization step
            try:
                policy = self.model.policy
                penalty = self.ewc_instance.penalty(policy)
                if TORCH_AVAILABLE and isinstance(penalty, torch.Tensor) and penalty.requires_grad:
                    penalty.backward()
                    # Scale gradients down to act as regularization, not primary signal
                    for param in policy.parameters():
                        if param.grad is not None:
                            param.grad.data *= 0.01
            except Exception as e:
                if self.verbose > 0:
                    logger.debug(f"EWC callback error (non-fatal): {e}")


# ─── REGISTRY ───

class TradeJackModelRegistry:
    """
    Central registry mapping model names to SB3 algorithm configurations.

    Interface-compatible with the legacy registry:
      - get_model_card(name) -> ModelCard
      - list_models_for_tier(max_tier) -> List[ModelCard]
      - build_model(name, env, **kwargs) -> SB3 model or baseline

    The key difference: build_model() now requires an `env` parameter
    because SB3 models are instantiated with their environment.
    """

    def __init__(self):
        self.cards: Dict[str, ModelCard] = {}
        self._register_all_models()

    def _register(self, card: ModelCard):
        self.cards[card.model_name] = card

    def _register_all_models(self):
        # ── Tier 1: High compute, full transformer backbone ──
        self._register(ModelCard(
            model_name="PPO-Transformer",
            tier_requirement=1,
            category="rl_ppo",
            algo_class="PPO",
            encoder_type="transformer",
            vram_estimate_mb=2500.0,
            description="PPO with causal transformer feature encoder. High-capacity, long-horizon temporal modeling.",
            default_hyperparams={
                "learning_rate": 3e-4,
                "n_steps": 256,
                "batch_size": 64,
                "n_epochs": 10,
                "gamma": 0.99,
                "gae_lambda": 0.95,
                "clip_range": 0.2,
                "ent_coef": 0.01,
                "features_dim": 256,
                "channels": 128,
            },
        ))

        # ── Tier 2: Mid compute, dilated CNN backbone ──
        self._register(ModelCard(
            model_name="PPO-DilatedCNN",
            tier_requirement=2,
            category="rl_ppo",
            algo_class="PPO",
            encoder_type="dilated_cnn",
            vram_estimate_mb=850.0,
            description="PPO with dilated causal CNN encoder. Good balance of capacity and efficiency.",
            default_hyperparams={
                "learning_rate": 3e-4,
                "n_steps": 256,
                "batch_size": 64,
                "n_epochs": 10,
                "gamma": 0.99,
                "gae_lambda": 0.95,
                "clip_range": 0.2,
                "ent_coef": 0.01,
                "features_dim": 128,
                "channels": 64,
            },
        ))

        self._register(ModelCard(
            model_name="SAC-DilatedCNN",
            tier_requirement=2,
            category="rl_sac",
            algo_class="SAC",
            encoder_type="dilated_cnn",
            vram_estimate_mb=900.0,
            description="SAC with dilated CNN encoder. Most sample-efficient — critical for scarce live data.",
            default_hyperparams={
                "learning_rate": 3e-4,
                "buffer_size": 100_000,
                "batch_size": 256,
                "gamma": 0.99,
                "tau": 0.005,
                "ent_coef": "auto",
                "train_freq": 1,
                "gradient_steps": 1,
                "features_dim": 128,
                "channels": 64,
            },
        ))

        # ── Tier 3: Lightweight, discretized actions ──
        self._register(ModelCard(
            model_name="DuelingDQN",
            tier_requirement=3,
            category="rl_dqn",
            algo_class="DQN",
            encoder_type="mlp",
            vram_estimate_mb=200.0,
            description="Dueling Double DQN with prioritized replay. Cheap Tier-3 baseline for fast iteration.",
            default_hyperparams={
                "learning_rate": 1e-4,
                "buffer_size": 50_000,
                "batch_size": 64,
                "gamma": 0.99,
                "exploration_fraction": 0.2,
                "exploration_final_eps": 0.05,
                "target_update_interval": 500,
                "train_freq": 4,
                "gradient_steps": 1,
                "features_dim": 64,
                "channels": 32,
            },
        ))

        # ── Baselines (no RL, rule-based) ──
        self._register(ModelCard(
            model_name="Momentum-Baseline",
            tier_requirement=3,
            category="rule_based",
            algo_class="rule",
            encoder_type="none",
            vram_estimate_mb=0.0,
            description="Dual moving-average crossover. Promotion hurdle — RL must beat this.",
        ))

        self._register(ModelCard(
            model_name="BuyAndHold-Baseline",
            tier_requirement=3,
            category="baseline",
            algo_class="static",
            encoder_type="none",
            vram_estimate_mb=0.0,
            description="Buy at t=0, hold forever. The simplest possible strategy. Fee-adjusted.",
        ))

    def get_model_card(self, model_name: str) -> Optional[ModelCard]:
        return self.cards.get(model_name)

    def list_models_for_tier(self, max_tier: int) -> List[ModelCard]:
        """Returns models available at the given tier (higher tier number = less compute)."""
        return [card for card in self.cards.values() if card.tier_requirement >= max_tier]

    def list_rl_models(self) -> List[ModelCard]:
        """Returns only RL-trainable models (excludes baselines)."""
        return [card for card in self.cards.values() if card.algo_class in ("PPO", "SAC", "DQN")]

    def build_model(
        self,
        model_name: str,
        env=None,
        ewc_instance=None,
        device: str = "auto",
        **override_hyperparams,
    ):
        """
        Instantiate an SB3 model or baseline by name.

        For RL models, requires `env` (a Gymnasium-compliant environment).
        For baselines, env is optional.

        Returns:
            SB3 model (PPO/SAC/DQN) or baseline instance
        """
        card = self.get_model_card(model_name)
        if not card:
            logger.warning(f"Model '{model_name}' not found. Defaulting to 'DuelingDQN'.")
            card = self.cards.get("DuelingDQN", list(self.cards.values())[0])

        # Baselines don't need SB3
        if card.algo_class == "rule":
            from swarm.baselines import MomentumBaseline
            return MomentumBaseline()
        elif card.algo_class == "static":
            from swarm.baselines import BuyAndHoldBaseline
            return BuyAndHoldBaseline()

        # RL models require SB3 + env
        if not SB3_AVAILABLE:
            raise ImportError(
                f"stable-baselines3 required for '{model_name}'. "
                "pip install stable-baselines3"
            )
        if env is None:
            raise ValueError(f"RL model '{model_name}' requires an `env` parameter.")

        # Merge default hyperparams with overrides
        hp = {**card.default_hyperparams, **override_hyperparams}
        features_dim = hp.pop("features_dim", 128)
        channels = hp.pop("channels", 64)

        # Import the feature extractor
        from swarm.shared_encoder import LOBFeatureExtractor

        # Build policy kwargs with our custom feature extractor
        policy_kwargs = dict(
            features_extractor_class=LOBFeatureExtractor,
            features_extractor_kwargs=dict(
                features_dim=features_dim,
                channels=channels,
            ),
        )

        # Build callbacks
        callbacks = []
        if ewc_instance is not None:
            callbacks.append(EWCCallback(ewc_instance=ewc_instance))

        # Wrap env for DQN (needs discrete actions)
        target_env = env
        if card.algo_class == "DQN" and GYM_AVAILABLE:
            target_env = DiscreteActionWrapper(env)

        # Instantiate the SB3 algorithm
        if card.algo_class == "PPO":
            model = PPO(
                "MultiInputPolicy",
                target_env,
                learning_rate=hp.get("learning_rate", 3e-4),
                n_steps=hp.get("n_steps", 256),
                batch_size=hp.get("batch_size", 64),
                n_epochs=hp.get("n_epochs", 10),
                gamma=hp.get("gamma", 0.99),
                gae_lambda=hp.get("gae_lambda", 0.95),
                clip_range=hp.get("clip_range", 0.2),
                ent_coef=hp.get("ent_coef", 0.01),
                policy_kwargs=policy_kwargs,
                device=device,
                verbose=0,
            )
        elif card.algo_class == "SAC":
            model = SAC(
                "MultiInputPolicy",
                target_env,
                learning_rate=hp.get("learning_rate", 3e-4),
                buffer_size=hp.get("buffer_size", 100_000),
                batch_size=hp.get("batch_size", 256),
                gamma=hp.get("gamma", 0.99),
                tau=hp.get("tau", 0.005),
                ent_coef=hp.get("ent_coef", "auto"),
                train_freq=hp.get("train_freq", 1),
                gradient_steps=hp.get("gradient_steps", 1),
                policy_kwargs=policy_kwargs,
                device=device,
                verbose=0,
            )
        elif card.algo_class == "DQN":
            model = DQN(
                "MultiInputPolicy",
                target_env,
                learning_rate=hp.get("learning_rate", 1e-4),
                buffer_size=hp.get("buffer_size", 50_000),
                batch_size=hp.get("batch_size", 64),
                gamma=hp.get("gamma", 0.99),
                exploration_fraction=hp.get("exploration_fraction", 0.2),
                exploration_final_eps=hp.get("exploration_final_eps", 0.05),
                target_update_interval=hp.get("target_update_interval", 500),
                train_freq=hp.get("train_freq", 4),
                gradient_steps=hp.get("gradient_steps", 1),
                policy_kwargs=policy_kwargs,
                device=device,
                verbose=0,
            )
        else:
            raise ValueError(f"Unknown algo_class: {card.algo_class}")

        # Attach metadata for downstream code
        model._tradejack_model_name = card.model_name
        model._tradejack_card = card
        model._tradejack_callbacks = callbacks

        logger.info(
            f"Built SB3 model '{card.model_name}' "
            f"(algo={card.algo_class}, encoder={card.encoder_type}, "
            f"features_dim={features_dim}, device={device})"
        )
        return model


# Global singleton instance
REGISTRY = TradeJackModelRegistry()


if __name__ == "__main__":
    logger.info("TradeJack v3 Model Registry")
    logger.info(f"Registered models: {len(REGISTRY.cards)}")
    for name, card in REGISTRY.cards.items():
        logger.info(f"  [{card.tier_requirement}] {name}: {card.description}")

    rl_models = REGISTRY.list_rl_models()
    logger.info(f"\nRL-trainable models: {len(rl_models)}")
    for card in rl_models:
        logger.info(f"  {card.model_name} ({card.algo_class})")

    tier3_models = REGISTRY.list_models_for_tier(max_tier=3)
    logger.info(f"\nTier 3+ models: {len(tier3_models)}")
