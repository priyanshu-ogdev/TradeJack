"""
Shared LOB Feature Encoder for all RL policy heads.

Provides two interfaces:
1. `LOBFeatureEncoder` — standalone nn.Module for direct use
2. `LOBFeatureExtractor` — SB3 BaseFeaturesExtractor wrapper for SB3 integration

Architecture: Dilated causal CNN (refactored from the legacy DilatedCNNSeq2SeqModel)
with a parallel portfolio-state MLP, concatenated into a single latent vector.

Input:
  lob_sequence: (batch, seq_len, 5) — [close, volume, ofi, vpin, kyles_lambda]
  portfolio_state: (batch, 4) — [cash_norm, position_qty, equity_norm, max_drawdown]

Output:
  (batch, features_dim) — latent representation fed to PPO/SAC/DQN policy heads
"""

import math
import logging
import numpy as np
from typing import Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import gymnasium as gym
    from gymnasium import spaces
    from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] (SharedEncoder) %(message)s")
logger = logging.getLogger("SharedEncoder")


class CausalDilatedBlock(nn.Module):
    """Single causal dilated convolution block with residual connection."""

    def __init__(self, channels: int, dilation: int, kernel_size: int = 3):
        super().__init__()
        # Causal padding: (kernel_size - 1) * dilation on the left only
        self.causal_pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(channels, channels, kernel_size=kernel_size, dilation=dilation)
        self.norm = nn.LayerNorm(channels)
        self.gate_conv = nn.Conv1d(channels, channels, kernel_size=kernel_size, dilation=dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, channels, seq_len)"""
        residual = x
        # Apply causal padding (left-pad only)
        padded = F.pad(x, (self.causal_pad, 0))
        h = torch.tanh(self.conv(padded))
        g = torch.sigmoid(self.gate_conv(padded))
        out = h * g
        # Residual connection
        return out + residual


class LOBFeatureEncoder(nn.Module):
    """
    Standalone dilated causal CNN encoder for LOB microstructure features.

    Takes raw (seq_len, 5) LOB features and produces a fixed-size latent vector.
    Can be pretrained supervised (predict next-tick return direction) before
    RL fine-tuning begins.
    """

    def __init__(
        self,
        lob_input_dim: int = 5,
        portfolio_input_dim: int = 4,
        channels: int = 64,
        dilations: tuple = (1, 2, 4, 8),
        features_dim: int = 128,
        **kwargs,
    ):
        super().__init__()
        if "lob_features" in kwargs:
            lob_input_dim = kwargs["lob_features"]
        if "portfolio_features" in kwargs:
            portfolio_input_dim = kwargs["portfolio_features"]
        if "hidden_dim" in kwargs:
            channels = kwargs["hidden_dim"]
        self.features_dim = features_dim

        # LOB sequence encoder: project input features to channel dim, then dilated CNN
        self.input_proj = nn.Conv1d(lob_input_dim, channels, kernel_size=1)
        self.dilated_blocks = nn.ModuleList([
            CausalDilatedBlock(channels, dilation=d) for d in dilations
        ])
        # Collapse sequence to single vector via adaptive pooling
        self.seq_pool = nn.AdaptiveAvgPool1d(1)
        self.seq_proj = nn.Linear(channels, features_dim // 2)

        # Portfolio state encoder: small MLP
        self.port_mlp = nn.Sequential(
            nn.Linear(portfolio_input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, features_dim // 2),
            nn.ReLU(),
        )

        # Final fusion
        self.fusion = nn.Sequential(
            nn.Linear(features_dim, features_dim),
            nn.ReLU(),
        )

    def forward(self, lob_sequence: torch.Tensor, portfolio_state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            lob_sequence: (batch, seq_len, lob_input_dim)
            portfolio_state: (batch, portfolio_input_dim)
        Returns:
            (batch, features_dim)
        """
        # LOB path: (batch, seq_len, C_in) -> (batch, C_in, seq_len) for Conv1d
        x = lob_sequence.transpose(1, 2)
        x = self.input_proj(x)
        for block in self.dilated_blocks:
            x = block(x)
        # Pool over sequence dimension
        x = self.seq_pool(x).squeeze(-1)  # (batch, channels)
        lob_features = self.seq_proj(x)  # (batch, features_dim // 2)

        # Portfolio path
        port_features = self.port_mlp(portfolio_state)  # (batch, features_dim // 2)

        # Concatenate and fuse
        combined = torch.cat([lob_features, port_features], dim=-1)
        return self.fusion(combined)


if SB3_AVAILABLE:
    class LOBFeatureExtractor(BaseFeaturesExtractor):
        """
        SB3-compatible feature extractor wrapping LOBFeatureEncoder.

        Designed for Dict observation spaces with keys:
          - "lob_sequence": Box(shape=(seq_len, 5))
          - "portfolio_state": Box(shape=(4,))

        Usage with SB3:
            model = PPO(
                "MultiInputPolicy", env,
                policy_kwargs=dict(
                    features_extractor_class=LOBFeatureExtractor,
                    features_extractor_kwargs=dict(features_dim=128),
                ),
            )
        """

        def __init__(self, observation_space: spaces.Dict, features_dim: int = 128,
                     channels: int = 64, dilations: tuple = (1, 2, 4, 8)):
            # Must call super().__init__ with the final features_dim
            super().__init__(observation_space, features_dim=features_dim)

            lob_shape = observation_space["lob_sequence"].shape  # (seq_len, 5)
            port_shape = observation_space["portfolio_state"].shape  # (4,)

            self.encoder = LOBFeatureEncoder(
                lob_input_dim=lob_shape[-1],
                portfolio_input_dim=port_shape[-1],
                channels=channels,
                dilations=dilations,
                features_dim=features_dim,
            )

        def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
            lob = observations["lob_sequence"]
            port = observations["portfolio_state"]
            return self.encoder(lob, port)

else:
    # Stub for environments without SB3
    class LOBFeatureExtractor:
        """Stub: SB3 not available. Install stable-baselines3 for full functionality."""
        def __init__(self, *args, **kwargs):
            raise ImportError("stable-baselines3 required for LOBFeatureExtractor. pip install stable-baselines3")


if __name__ == "__main__":
    logger.info("Testing LOBFeatureEncoder standalone...")
    encoder = LOBFeatureEncoder(lob_input_dim=5, portfolio_input_dim=4, features_dim=128)
    lob = torch.randn(4, 64, 5)
    port = torch.randn(4, 4)
    out = encoder(lob, port)
    logger.info(f"Encoder output shape: {out.shape}")  # Expected: (4, 128)
    assert out.shape == (4, 128), f"Expected (4, 128), got {out.shape}"

    total_params = sum(p.numel() for p in encoder.parameters())
    logger.info(f"Total parameters: {total_params:,}")
    logger.info("LOBFeatureEncoder test passed.")
