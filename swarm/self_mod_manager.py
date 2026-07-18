"""
Self-Modification Engine (`SelfModEngine` for `automaton` `self-mod/`).
Dynamically modifies and swaps model architectures from `Stock-Prediction-Models` (`Attention-is-all-you-Need`,
`Dilated-CNN-Seq2seq`, `neuro-evolution-novelty-search`, `Deep-Q-learning`) depending on Warden VRAM tier allocation.
Automatically injects Grace Blackwell FP8 mixed precision (`torch.float8_e4m3fn`) and Elastic Weight Consolidation (EWC).
"""

import os
import sys
import time
import json
import logging
import inspect
import math
import numpy as np
from typing import Dict, Any, List, Optional, Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] (SelfModEngine) %(message)s")
logger = logging.getLogger("SelfModEngine")

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.info("PyTorch not installed; SelfModEngine using Numpy simulation templates.")

from swarm.ewc_optimizer import ElasticWeightConsolidation

# ─── IMMUTABLE SAFETY INVARIANTS (Conway Automaton self-mod/code.ts pattern) ───
PROTECTED_FILES: frozenset = frozenset([
    "warden/warden_core.py",
    "escrow/escrow_contract.py",
    "swarm/self_mod_manager.py",
    "state/ledger.sqlite",
    "README.md",
    "constitution.md",
    "wallet.json",
    "agent/policy-engine.ts"
])

MAX_MODIFICATION_SIZE = 100_000  # 100KB max diff size


def validate_self_mod_safety(file_path: str, content_size: int = 0) -> Tuple[bool, str]:
    """
    Validates proposed code or weight modification against hard-coded safety invariants.
    Inspired by Conway-Research/automaton trust boundary architecture.
    """
    normalized_path = os.path.normpath(file_path).replace("\\", "/")
    for protected in PROTECTED_FILES:
        if normalized_path.endswith(protected) or normalized_path == protected:
            return False, f"BLOCKED: Cannot modify protected file '{file_path}'. Hard-coded safety invariant."
    if content_size > MAX_MODIFICATION_SIZE:
        return False, f"BLOCKED: Content size ({content_size} bytes) exceeds maximum ({MAX_MODIFICATION_SIZE} bytes)."
    return True, "All safety checks passed."


if TORCH_AVAILABLE:
    class PositionalEncoding(nn.Module):
        def __init__(self, d_model: int, max_len: int = 5000):
            super().__init__()
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(position * div_term)
            if d_model % 2 == 1:
                pe[:, 1::2] = torch.cos(position * div_term[:-1])
            else:
                pe[:, 1::2] = torch.cos(position * div_term)
            self.register_buffer('pe', pe.unsqueeze(0))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x + self.pe[:, :x.size(1), :]


class AttentionIsAllYouNeedTemplate:
    """
    Positional-Encoded Causal Transformer (`Stock-Prediction-Models/models/transformers/Attention-is-all-you-Need`).
    Features multi-head self-attention, sinusoidal positional encoding, and causal sequence masking.
    """
    def __init__(self, input_dim: int = 8, d_model: int = 128, nhead: int = 8, num_layers: int = 4):
        self.model_name = "Attention-is-all-you-Need"
        self.input_dim = input_dim
        self.d_model = d_model
        if TORCH_AVAILABLE:
            self.input_proj = nn.Linear(input_dim, d_model)
            self.pos_encoder = PositionalEncoding(d_model)
            encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_model*4, batch_first=True)
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.output_proj = nn.Linear(d_model, 1)
            self.net = nn.Sequential(self.input_proj, self.pos_encoder, self.transformer, self.output_proj)
        else:
            self.weights = {
                "encoder": np.random.normal(0, 0.05, (input_dim, d_model)).astype(np.float32),
                "decoder": np.random.normal(0, 0.05, (d_model, 1)).astype(np.float32)
            }

    def forward(self, x: Any) -> Any:
        if TORCH_AVAILABLE and hasattr(self, "input_proj"):
            if x.dim() == 2:
                x = x.unsqueeze(0)
            seq_len = x.size(1)
            # Generate causal mask to prevent attending to future order book states
            mask = torch.triu(torch.full((seq_len, seq_len), float('-inf'), device=x.device), diagonal=1)
            h = self.input_proj(x)
            h = self.pos_encoder(h)
            out = self.transformer(h, mask=mask)
            return self.output_proj(out)
        return np.sum(x, axis=-1, keepdims=True) * 0.01


class DilatedCNNSeq2SeqTemplate:
    """
    Stacked Residual Causal Dilated CNN (`Stock-Prediction-Models/models/cnn/Dilated-CNN-Seq2seq`).
    Features exponential dilation rates [1, 2, 4, 8] with causal left-padding and residual connections.
    """
    def __init__(self, input_dim: int = 8, channels: int = 64):
        self.model_name = "Dilated-CNN-Seq2seq"
        self.input_dim = input_dim
        self.channels = channels
        self.dilations = [1, 2, 4, 8]
        if TORCH_AVAILABLE:
            self.input_conv = nn.Conv1d(input_dim, channels, kernel_size=1)
            self.conv_blocks = nn.ModuleList([
                nn.Conv1d(channels, channels, kernel_size=3, dilation=d, padding=2*d)
                for d in self.dilations
            ])
            self.fc = nn.Linear(channels, 1)
            self.net = nn.ModuleList([self.input_conv, self.conv_blocks, self.fc])
        else:
            self.weights = {"conv": np.random.normal(0, 0.05, (input_dim, channels)).astype(np.float32)}

    def forward(self, x: Any) -> Any:
        if TORCH_AVAILABLE and hasattr(self, "conv_blocks"):
            if x.dim() == 3:
                x = x.transpose(1, 2)
            elif x.dim() == 2:
                x = x.unsqueeze(0).transpose(1, 2)
            h = self.input_conv(x)
            for conv, d in zip(self.conv_blocks, self.dilations):
                residual = h
                out = torch.relu(conv(h))
                # Slice causal padding from right
                out = out[:, :, :-2*d] if 2*d > 0 else out
                h = torch.relu(out + residual)
            pooled = torch.mean(h, dim=2)
            return self.fc(pooled)
        return np.sum(x, axis=-1, keepdims=True) * 0.008


class DeepQLearningTemplate:
    """
    Dueling Q-Network & Neuro-Evolution Novelty Scalper (`Stock-Prediction-Models/models/rl/Deep-Q-learning`).
    Splits Q-value estimation into independent Value V(s) and Advantage A(s, a) streams.
    """
    def __init__(self, input_dim: int = 8, hidden_dim: int = 32):
        self.model_name = "Deep-Q-learning"
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        if TORCH_AVAILABLE:
            self.fc_feat = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU())
            self.fc_val = nn.Linear(hidden_dim, 1)
            self.fc_adv = nn.Linear(hidden_dim, 3)  # 0: Flat, 1: Long, 2: Short
            self.net = nn.ModuleList([self.fc_feat, self.fc_val, self.fc_adv])
        else:
            self.weights = {
                "val": np.random.normal(0, 0.05, (input_dim, 1)).astype(np.float32),
                "adv": np.random.normal(0, 0.05, (input_dim, 3)).astype(np.float32)
            }

    def forward(self, x: Any) -> Any:
        if TORCH_AVAILABLE and hasattr(self, "fc_feat"):
            if x.dim() == 3:
                x = torch.mean(x, dim=1)
            elif x.dim() == 1:
                x = x.unsqueeze(0)
            feat = self.fc_feat(x)
            val = self.fc_val(feat)
            adv = self.fc_adv(feat)
            q_values = val + adv - torch.mean(adv, dim=1, keepdim=True)
            return q_values
        return np.array([0.1, 0.5, -0.2], dtype=np.float32)

    def mutate_novelty(self, mutation_rate: float = 0.05) -> None:
        """Applies Neuro-Evolution Novelty Search parameter perturbation."""
        if TORCH_AVAILABLE and hasattr(self, "net"):
            for param in self.net.parameters():
                if torch.rand(1).item() < 0.5:
                    noise = torch.randn_like(param.data) * mutation_rate
                    param.data.add_(noise)
        elif hasattr(self, "weights"):
            for k in self.weights:
                self.weights[k] += np.random.normal(0, mutation_rate, self.weights[k].shape).astype(np.float32)


class SelfModEngine:
    """
    Manages self-modification and model architecture transitions inside Docker.
    """

    def __init__(self, child_id: int = 0, state_dir: str = "d:/TradeJack/state"):
        self.child_id = child_id
        self.state_dir = os.path.abspath(state_dir)
        self.active_model: Any = DilatedCNNSeq2SeqTemplate()
        self.ewc_instance: Optional[ElasticWeightConsolidation] = None
        self.active_tier: int = 2

    def inject_blackwell_fp8_optimizations(self, code_string: str) -> str:
        """
        Dynamically rewrites generated PyTorch code strings to inject Grace Blackwell FP8 mixed precision
        and gradient checkpointing.
        """
        if "torch.cuda.amp.autocast" not in code_string:
            injection = (
                "# [SelfMod Engine] Injected Grace Blackwell Native FP8 Autocast\n"
                "if torch.cuda.is_available() and hasattr(torch, 'float8_e4m3fn'):\n"
                "    autocast_context = torch.cuda.amp.autocast(dtype=torch.float8_e4m3fn)\n"
                "else:\n"
                "    autocast_context = torch.cuda.amp.autocast(dtype=torch.float16)\n"
            )
            code_string = injection + "\n" + code_string
        return code_string

    def modify_source_file(self, file_path: str, new_content: str, reason: str) -> Dict[str, Any]:
        """
        Safely attempts to edit a source file after verifying against hard-coded safety invariants (`PROTECTED_FILES`).
        Adapted from Conway-Research/automaton (`self-mod/code.ts`).
        """
        allowed, msg = validate_self_mod_safety(file_path, len(new_content))
        if not allowed:
            logger.warning(f"SelfMod attempt blocked by guardrails on '{file_path}': {msg}")
            return {"success": False, "error": msg, "reason": reason}
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(new_content)
            logger.info(f"SelfMod successfully modified '{file_path}'. Reason: {reason}")
            return {"success": True, "reason": reason}
        except Exception as e:
            return {"success": False, "error": str(e), "reason": reason}

    def swap_active_architecture(
        self,
        target_model_name: str,
        current_tier: int,
        calibration_loader: Optional[Any] = None
    ) -> Any:
        """
        Enforces Warden memory limits and swaps model architecture while computing EWC Fisher bounds
        to prevent Catastrophic Forgetting.
        """
        self.active_tier = current_tier
        logger.info(f"SelfMod Engine triggered model swap to '{target_model_name}' under Tier {current_tier} rules.")
        
        # Enforce Warden limits
        if current_tier == 3 and target_model_name != "Deep-Q-learning":
            logger.warning(f"Tier 3 (Inference-Only) forbids '{target_model_name}'. Forcing 'Deep-Q-learning' scalping model.")
            target_model_name = "Deep-Q-learning"
        elif current_tier == 2 and target_model_name == "Attention-is-all-you-Need":
            logger.warning("Tier 2 (4GB VRAM) forbids heavy 'Attention-is-all-you-Need'. Forcing 'Dilated-CNN-Seq2seq'.")
            target_model_name = "Dilated-CNN-Seq2seq"
            
        # Compute EWC on outgoing active model before destroying or offloading
        if self.active_model is not None and calibration_loader is not None:
            logger.info("Computing EWC Fisher matrix on outgoing model...")
            self.ewc_instance = ElasticWeightConsolidation(self.active_model, calibration_loader)
            
        # Instantiate new architecture
        if target_model_name == "Attention-is-all-you-Need":
            new_model = AttentionIsAllYouNeedTemplate()
        elif target_model_name == "Deep-Q-learning":
            new_model = DeepQLearningTemplate()
        else:
            new_model = DilatedCNNSeq2SeqTemplate()
            
        self.active_model = new_model
        return new_model


if __name__ == "__main__":
    logger.info("Testing SelfModEngine standalone execution...")
    engine = SelfModEngine(child_id=1)
    print("Initial active model:", engine.active_model.model_name)
    
    # Test tier 2 restriction
    engine.swap_active_architecture("Attention-is-all-you-Need", current_tier=2)
    print("Active model after Tier 2 swap request:", engine.active_model.model_name)
    
    # Test tier 1 allowance
    engine.swap_active_architecture("Attention-is-all-you-Need", current_tier=1)
    print("Active model after Tier 1 swap request:", engine.active_model.model_name)
