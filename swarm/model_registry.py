"""
Universal DGX Model Registry & Sovereign Swarm Architecture Library (`TradeJackModelRegistry`).
Fuses the complete model pool of `huseinzol05/Stock-Prediction-Models` (`deep-learning`, `agent`, `stacking`)
with the memory quota and policy card specifications of `Conway-Research/automaton` (`registry/agent-card.ts`).

Enforces Grace Blackwell Unified Memory VRAM tiers:
- Tier 1: High Alpha / Deep Ensemble (<= 20GB VRAM)
- Tier 2: Mid Compute / Standard Sequence & Actor-Critic (<= 4GB VRAM)
- Tier 3: Critical Survival / Micro-Scalper & Neuro-Evolution (<= 1GB / CPU/FP8)
"""

import os
import sys
import math
import time
import logging
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Callable, Tuple, Union

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] (ModelRegistry) %(message)s")
logger = logging.getLogger("ModelRegistry")

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not installed; ModelRegistry operating in Laptop Simulation Mode (CPU / Numpy fallback).")


@dataclass
class ModelCard:
    """
    Agent/Model Policy Specification Card matching Conway Automaton `registry/agent-card.ts`.
    """
    model_name: str
    tier_requirement: int  # 1 (20GB), 2 (4GB), or 3 (1GB/Inference)
    category: str          # transformer, seq2seq_vae, cnn, rnn, actor_critic, q_learning, neuro_evolution, ensemble
    vram_estimate_mb: float
    description: str
    builder_fn: Callable[..., Any]


# ─── HELPER MODULES (PYTORCH) ───
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
else:
    class PositionalEncoding:
        def __init__(self, *args, **kwargs): pass
        def forward(self, x: Any) -> Any: return x


# ─── TIER 1 ARCHITECTURES: HIGH ALPHA / DEEP ENSEMBLES (<= 20GB VRAM) ───

class AttentionIsAllYouNeedModel:
    """Positional-Encoded Causal Transformer (`Stock-Prediction-Models/deep-learning/attention-is-all-you-need`)."""
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
            self.weights = {"encoder": np.random.normal(0, 0.05, (input_dim, d_model)).astype(np.float32), "decoder": np.random.normal(0, 0.05, (d_model, 1)).astype(np.float32)}

    def forward(self, x: Any) -> Any:
        if TORCH_AVAILABLE and hasattr(self, "input_proj"):
            if x.dim() == 2: x = x.unsqueeze(0)
            seq_len = x.size(1)
            mask = torch.triu(torch.full((seq_len, seq_len), float('-inf'), device=x.device), diagonal=1)
            h = self.pos_encoder(self.input_proj(x))
            out = self.transformer(h, mask=mask)
            return self.output_proj(out)
        return np.sum(x, axis=-1, keepdims=True) * 0.01


class LSTMSeq2SeqVAEModel:
    """LSTM Sequence-to-Sequence Variational Autoencoder (`deep-learning/lstm-seq2seq-vae`)."""
    def __init__(self, input_dim: int = 8, hidden_dim: int = 64, latent_dim: int = 16):
        self.model_name = "LSTM-Seq2Seq-VAE"
        self.input_dim = input_dim
        if TORCH_AVAILABLE:
            self.encoder_lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
            self.fc_mu = nn.Linear(hidden_dim, latent_dim)
            self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
            self.fc_latent_to_dec = nn.Linear(latent_dim, hidden_dim)
            self.decoder_lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
            self.output_head = nn.Linear(hidden_dim, 1)
            self.net = nn.ModuleList([self.encoder_lstm, self.fc_mu, self.fc_logvar, self.fc_latent_to_dec, self.decoder_lstm, self.output_head])
        else:
            self.weights = {"vae": np.random.normal(0, 0.05, (input_dim, 1)).astype(np.float32)}

    def forward(self, x: Any) -> Any:
        if TORCH_AVAILABLE and hasattr(self, "encoder_lstm"):
            if x.dim() == 2: x = x.unsqueeze(0)
            _, (h_n, _) = self.encoder_lstm(x)
            h_last = h_n[-1]
            mu = self.fc_mu(h_last)
            logvar = self.fc_logvar(h_last)
            # Reparameterization
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + eps * std
            dec_init = torch.relu(self.fc_latent_to_dec(z)).unsqueeze(0)
            dec_out, _ = self.decoder_lstm(x, (dec_init, torch.zeros_like(dec_init)))
            return self.output_head(dec_out[:, -1, :])
        return np.sum(x, axis=-1, keepdims=True) * 0.012


class GRUSeq2SeqVAEModel:
    """GRU Sequence-to-Sequence Variational Autoencoder (`deep-learning/gru-seq2seq-vae`)."""
    def __init__(self, input_dim: int = 8, hidden_dim: int = 64, latent_dim: int = 16):
        self.model_name = "GRU-Seq2Seq-VAE"
        self.input_dim = input_dim
        if TORCH_AVAILABLE:
            self.encoder_gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
            self.fc_mu = nn.Linear(hidden_dim, latent_dim)
            self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
            self.fc_latent_to_dec = nn.Linear(latent_dim, hidden_dim)
            self.decoder_gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
            self.output_head = nn.Linear(hidden_dim, 1)
            self.net = nn.ModuleList([self.encoder_gru, self.fc_mu, self.fc_logvar, self.fc_latent_to_dec, self.decoder_gru, self.output_head])
        else:
            self.weights = {"vae": np.random.normal(0, 0.05, (input_dim, 1)).astype(np.float32)}

    def forward(self, x: Any) -> Any:
        if TORCH_AVAILABLE and hasattr(self, "encoder_gru"):
            if x.dim() == 2: x = x.unsqueeze(0)
            _, h_n = self.encoder_gru(x)
            h_last = h_n[-1]
            mu = self.fc_mu(h_last)
            logvar = self.fc_logvar(h_last)
            std = torch.exp(0.5 * logvar)
            z = mu + torch.randn_like(std) * std
            dec_init = torch.relu(self.fc_latent_to_dec(z)).unsqueeze(0)
            dec_out, _ = self.decoder_gru(x, dec_init)
            return self.output_head(dec_out[:, -1, :])
        return np.sum(x, axis=-1, keepdims=True) * 0.011


class StackingEncoderEnsembleModel:
    """Stacking Autoencoder & Multi-Path Classifier Ensemble (`stacking/stack-encoder-ensemble-xgb`)."""
    def __init__(self, input_dim: int = 8, hidden_dim: int = 32):
        self.model_name = "Stacking-Encoder-Ensemble"
        self.input_dim = input_dim
        if TORCH_AVAILABLE:
            self.ae_enc = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim // 2))
            self.path_conv = nn.Conv1d(input_dim, 16, kernel_size=3, padding=1)
            self.path_lstm = nn.LSTM(input_dim, 16, batch_first=True)
            self.ensemble_head = nn.Linear((hidden_dim // 2) + 16 + 16, 1)
            self.net = nn.ModuleList([self.ae_enc, self.path_conv, self.path_lstm, self.ensemble_head])
        else:
            self.weights = {"ens": np.random.normal(0, 0.05, (input_dim, 1)).astype(np.float32)}

    def forward(self, x: Any) -> Any:
        if TORCH_AVAILABLE and hasattr(self, "ae_enc"):
            if x.dim() == 2: x = x.unsqueeze(0)
            ae_feat = self.ae_enc(x[:, -1, :])
            conv_feat = torch.mean(torch.relu(self.path_conv(x.transpose(1, 2))), dim=2)
            _, (lstm_h, _) = self.path_lstm(x)
            lstm_feat = lstm_h[-1]
            combined = torch.cat([ae_feat, conv_feat, lstm_feat], dim=1)
            return self.ensemble_head(combined)
        return np.sum(x, axis=-1, keepdims=True) * 0.015


class StackRNNARIMAXGBModel:
    """Hybrid RNN + Statistical Momentum Stacking Model (`stacking/stack-rnn-arima-xgb`)."""
    def __init__(self, input_dim: int = 8, hidden_dim: int = 32):
        self.model_name = "Stack-RNN-ARIMA-XGB"
        self.input_dim = input_dim
        if TORCH_AVAILABLE:
            self.rnn = nn.GRU(input_dim, hidden_dim, batch_first=True)
            self.stat_proj = nn.Linear(input_dim, hidden_dim // 2)
            self.fc = nn.Linear(hidden_dim + (hidden_dim // 2), 1)
            self.net = nn.ModuleList([self.rnn, self.stat_proj, self.fc])
        else:
            self.weights = {"hybrid": np.random.normal(0, 0.05, (input_dim, 1)).astype(np.float32)}

    def forward(self, x: Any) -> Any:
        if TORCH_AVAILABLE and hasattr(self, "rnn"):
            if x.dim() == 2: x = x.unsqueeze(0)
            _, h_n = self.rnn(x)
            stat_feat = torch.relu(self.stat_proj(torch.mean(x, dim=1)))
            combined = torch.cat([h_n[-1], stat_feat], dim=1)
            return self.fc(combined)
        return np.sum(x, axis=-1, keepdims=True) * 0.014


# ─── TIER 2 ARCHITECTURES: MID COMPUTE / STANDARD SEQUENCE & ACTOR-CRITIC (<= 4GB VRAM) ───

class DilatedCNNSeq2SeqModel:
    """Stacked Residual Causal Dilated CNN (`deep-learning/dilated-cnn-seq2seq`)."""
    def __init__(self, input_dim: int = 8, channels: int = 64):
        self.model_name = "Dilated-CNN-Seq2seq"
        self.input_dim = input_dim
        self.channels = channels
        self.dilations = [1, 2, 4, 8]
        if TORCH_AVAILABLE:
            self.input_conv = nn.Conv1d(input_dim, channels, kernel_size=1)
            self.conv_blocks = nn.ModuleList([nn.Conv1d(channels, channels, kernel_size=3, dilation=d, padding=2*d) for d in self.dilations])
            self.fc = nn.Linear(channels, 1)
            self.net = nn.ModuleList([self.input_conv, self.conv_blocks, self.fc])
        else:
            self.weights = {"conv": np.random.normal(0, 0.05, (input_dim, channels)).astype(np.float32)}

    def forward(self, x: Any) -> Any:
        if TORCH_AVAILABLE and hasattr(self, "conv_blocks"):
            if x.dim() == 3: x = x.transpose(1, 2)
            elif x.dim() == 2: x = x.unsqueeze(0).transpose(1, 2)
            h = self.input_conv(x)
            for conv, d in zip(self.conv_blocks, self.dilations):
                res = h
                out = torch.relu(conv(h))
                out = out[:, :, :-2*d] if 2*d > 0 else out
                h = torch.relu(out + res)
            return self.fc(torch.mean(h, dim=2))
        return np.sum(x, axis=-1, keepdims=True) * 0.008


class CNNSeq2SeqModel:
    """Causal 1D Convolutional Sequence Model (`deep-learning/cnn-seq2seq`)."""
    def __init__(self, input_dim: int = 8, channels: int = 32):
        self.model_name = "CNN-Seq2seq"
        self.input_dim = input_dim
        if TORCH_AVAILABLE:
            self.conv1 = nn.Conv1d(input_dim, channels, kernel_size=3, padding=2)
            self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=2)
            self.fc = nn.Linear(channels, 1)
            self.net = nn.Sequential(self.conv1, nn.ReLU(), self.conv2, nn.ReLU())
        else:
            self.weights = {"cnn": np.random.normal(0, 0.05, (input_dim, 1)).astype(np.float32)}

    def forward(self, x: Any) -> Any:
        if TORCH_AVAILABLE and hasattr(self, "conv1"):
            if x.dim() == 3: x = x.transpose(1, 2)
            elif x.dim() == 2: x = x.unsqueeze(0).transpose(1, 2)
            h = torch.relu(self.conv1(x)[:, :, :-2])
            h = torch.relu(self.conv2(h)[:, :, :-2])
            return self.fc(torch.mean(h, dim=2))
        return np.sum(x, axis=-1, keepdims=True) * 0.009


class LSTMSeq2SeqModel:
    """Standard LSTM Sequence-to-Sequence (`deep-learning/lstm-seq2seq`)."""
    def __init__(self, input_dim: int = 8, hidden_dim: int = 64):
        self.model_name = "LSTM-Seq2seq"
        self.input_dim = input_dim
        if TORCH_AVAILABLE:
            self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=2, batch_first=True)
            self.fc = nn.Linear(hidden_dim, 1)
            self.net = nn.ModuleList([self.lstm, self.fc])
        else:
            self.weights = {"lstm": np.random.normal(0, 0.05, (input_dim, 1)).astype(np.float32)}

    def forward(self, x: Any) -> Any:
        if TORCH_AVAILABLE and hasattr(self, "lstm"):
            if x.dim() == 2: x = x.unsqueeze(0)
            out, _ = self.lstm(x)
            return self.fc(out[:, -1, :])
        return np.sum(x, axis=-1, keepdims=True) * 0.010


class BiLSTMSeq2SeqModel:
    """Bidirectional LSTM Sequence Model (`deep-learning/bidirectional-lstm-seq2seq`)."""
    def __init__(self, input_dim: int = 8, hidden_dim: int = 64):
        self.model_name = "Bidirectional-LSTM-Seq2seq"
        self.input_dim = input_dim
        if TORCH_AVAILABLE:
            self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=2, batch_first=True, bidirectional=True)
            self.fc = nn.Linear(hidden_dim * 2, 1)
            self.net = nn.ModuleList([self.lstm, self.fc])
        else:
            self.weights = {"bilstm": np.random.normal(0, 0.05, (input_dim, 1)).astype(np.float32)}

    def forward(self, x: Any) -> Any:
        if TORCH_AVAILABLE and hasattr(self, "lstm"):
            if x.dim() == 2: x = x.unsqueeze(0)
            out, _ = self.lstm(x)
            return self.fc(out[:, -1, :])
        return np.sum(x, axis=-1, keepdims=True) * 0.011


class GRUSeq2SeqModel:
    """Standard GRU Sequence-to-Sequence (`deep-learning/gru-seq2seq`)."""
    def __init__(self, input_dim: int = 8, hidden_dim: int = 64):
        self.model_name = "GRU-Seq2seq"
        self.input_dim = input_dim
        if TORCH_AVAILABLE:
            self.gru = nn.GRU(input_dim, hidden_dim, num_layers=2, batch_first=True)
            self.fc = nn.Linear(hidden_dim, 1)
            self.net = nn.ModuleList([self.gru, self.fc])
        else:
            self.weights = {"gru": np.random.normal(0, 0.05, (input_dim, 1)).astype(np.float32)}

    def forward(self, x: Any) -> Any:
        if TORCH_AVAILABLE and hasattr(self, "gru"):
            if x.dim() == 2: x = x.unsqueeze(0)
            out, _ = self.gru(x)
            return self.fc(out[:, -1, :])
        return np.sum(x, axis=-1, keepdims=True) * 0.009


class VanillaSeq2SeqModel:
    """Vanilla RNN Sequence-to-Sequence (`deep-learning/vanilla`)."""
    def __init__(self, input_dim: int = 8, hidden_dim: int = 32):
        self.model_name = "Vanilla-Seq2seq"
        self.input_dim = input_dim
        if TORCH_AVAILABLE:
            self.rnn = nn.RNN(input_dim, hidden_dim, num_layers=2, batch_first=True)
            self.fc = nn.Linear(hidden_dim, 1)
            self.net = nn.ModuleList([self.rnn, self.fc])
        else:
            self.weights = {"rnn": np.random.normal(0, 0.05, (input_dim, 1)).astype(np.float32)}

    def forward(self, x: Any) -> Any:
        if TORCH_AVAILABLE and hasattr(self, "rnn"):
            if x.dim() == 2: x = x.unsqueeze(0)
            out, _ = self.rnn(x)
            return self.fc(out[:, -1, :])
        return np.sum(x, axis=-1, keepdims=True) * 0.007


class ActorCriticAgentModel:
    """Actor-Critic Policy & Value Network (`agent/actor-critic-agent`)."""
    def __init__(self, input_dim: int = 8, hidden_dim: int = 32):
        self.model_name = "Actor-Critic-Agent"
        self.input_dim = input_dim
        if TORCH_AVAILABLE:
            self.fc_shared = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU())
            self.actor_head = nn.Linear(hidden_dim, 3)  # Flat, Long, Short
            self.critic_head = nn.Linear(hidden_dim, 1) # Value V(s)
            self.net = nn.ModuleList([self.fc_shared, self.actor_head, self.critic_head])
        else:
            self.weights = {"actor": np.random.normal(0, 0.05, (input_dim, 3)).astype(np.float32)}

    def forward(self, x: Any) -> Any:
        if TORCH_AVAILABLE and hasattr(self, "fc_shared"):
            if x.dim() == 3: x = torch.mean(x, dim=1)
            elif x.dim() == 1: x = x.unsqueeze(0)
            feat = self.fc_shared(x)
            logits = self.actor_head(feat)
            val = self.critic_head(feat)
            # Return combined advantage-weighted score for think loop
            return val + (logits[:, 1] - logits[:, 2]).unsqueeze(-1)
        return np.array([0.1, 0.6, -0.1], dtype=np.float32)


class ActorCriticDuelAgentModel:
    """Dueling Advantage Actor-Critic (`agent/actor-critic-duel-agent`)."""
    def __init__(self, input_dim: int = 8, hidden_dim: int = 32):
        self.model_name = "Actor-Critic-Duel-Agent"
        self.input_dim = input_dim
        if TORCH_AVAILABLE:
            self.fc_feat = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU())
            self.val_head = nn.Linear(hidden_dim, 1)
            self.adv_head = nn.Linear(hidden_dim, 3)
            self.net = nn.ModuleList([self.fc_feat, self.val_head, self.adv_head])
        else:
            self.weights = {"duel": np.random.normal(0, 0.05, (input_dim, 3)).astype(np.float32)}

    def forward(self, x: Any) -> Any:
        if TORCH_AVAILABLE and hasattr(self, "fc_feat"):
            if x.dim() == 3: x = torch.mean(x, dim=1)
            elif x.dim() == 1: x = x.unsqueeze(0)
            feat = self.fc_feat(x)
            val = self.val_head(feat)
            adv = self.adv_head(feat)
            return val + adv - torch.mean(adv, dim=1, keepdim=True)
        return np.array([0.1, 0.7, -0.2], dtype=np.float32)


class ActorCriticRecurrentAgentModel:
    """Recurrent LSTM Actor-Critic (`agent/actor-critic-recurrent-agent`)."""
    def __init__(self, input_dim: int = 8, hidden_dim: int = 32):
        self.model_name = "Actor-Critic-Recurrent-Agent"
        self.input_dim = input_dim
        if TORCH_AVAILABLE:
            self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
            self.actor = nn.Linear(hidden_dim, 3)
            self.critic = nn.Linear(hidden_dim, 1)
            self.net = nn.ModuleList([self.lstm, self.actor, self.critic])
        else:
            self.weights = {"rec_ac": np.random.normal(0, 0.05, (input_dim, 3)).astype(np.float32)}

    def forward(self, x: Any) -> Any:
        if TORCH_AVAILABLE and hasattr(self, "lstm"):
            if x.dim() == 2: x = x.unsqueeze(0)
            _, (h_n, _) = self.lstm(x)
            h = h_n[-1]
            return self.critic(h) + (self.actor(h)[:, 1] - self.actor(h)[:, 2]).unsqueeze(-1)
        return np.array([0.1, 0.5, -0.1], dtype=np.float32)


class DoubleDuelRecurrentQAgentModel:
    """Double Dueling Recurrent Q-Learning Agent (`agent/double-duel-recurrent-q-learning-agent`)."""
    def __init__(self, input_dim: int = 8, hidden_dim: int = 32):
        self.model_name = "Double-Duel-Recurrent-Q-Learning-Agent"
        self.input_dim = input_dim
        if TORCH_AVAILABLE:
            self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
            self.val = nn.Linear(hidden_dim, 1)
            self.adv = nn.Linear(hidden_dim, 3)
            self.net = nn.ModuleList([self.gru, self.val, self.adv])
        else:
            self.weights = {"ddrq": np.random.normal(0, 0.05, (input_dim, 3)).astype(np.float32)}

    def forward(self, x: Any) -> Any:
        if TORCH_AVAILABLE and hasattr(self, "gru"):
            if x.dim() == 2: x = x.unsqueeze(0)
            _, h_n = self.gru(x)
            h = h_n[-1]
            adv = self.adv(h)
            return self.val(h) + adv - torch.mean(adv, dim=1, keepdim=True)
        return np.array([0.2, 0.6, -0.3], dtype=np.float32)


# ─── TIER 3 ARCHITECTURES: CRITICAL SURVIVAL / SCALPERS & NEURO-EVOLUTION (<= 1GB / CPU/FP8) ───

class DeepQLearningModel:
    """Dueling Q-Network Micro-Scalper (`agent/duel-q-learning-agent` / `Deep-Q-learning`)."""
    def __init__(self, input_dim: int = 8, hidden_dim: int = 32):
        self.model_name = "Deep-Q-learning"
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        if TORCH_AVAILABLE:
            self.fc_feat = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU())
            self.fc_val = nn.Linear(hidden_dim, 1)
            self.fc_adv = nn.Linear(hidden_dim, 3)
            self.net = nn.ModuleList([self.fc_feat, self.fc_val, self.fc_adv])
        else:
            self.weights = {
                "val": np.random.normal(0, 0.05, (input_dim, 1)).astype(np.float32),
                "adv": np.random.normal(0, 0.05, (input_dim, 3)).astype(np.float32)
            }

    def forward(self, x: Any) -> Any:
        if TORCH_AVAILABLE and hasattr(self, "fc_feat"):
            if x.dim() == 3: x = torch.mean(x, dim=1)
            elif x.dim() == 1: x = x.unsqueeze(0)
            feat = self.fc_feat(x)
            adv = self.fc_adv(feat)
            return self.fc_val(feat) + adv - torch.mean(adv, dim=1, keepdim=True)
        return np.array([0.1, 0.5, -0.2], dtype=np.float32)

    def mutate_novelty(self, mutation_rate: float = 0.05) -> None:
        if TORCH_AVAILABLE and hasattr(self, "net"):
            for param in self.net.parameters():
                if torch.rand(1).item() < 0.5:
                    param.data.add_(torch.randn_like(param.data) * mutation_rate)
        elif hasattr(self, "weights"):
            for k in self.weights:
                self.weights[k] += np.random.normal(0, mutation_rate, self.weights[k].shape).astype(np.float32)


class DoubleQLearningModel:
    """Double Q-Learning Agent (`agent/double-q-learning-agent`)."""
    def __init__(self, input_dim: int = 8, hidden_dim: int = 16):
        self.model_name = "Double-Q-learning-Agent"
        self.input_dim = input_dim
        if TORCH_AVAILABLE:
            self.net = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 3))
        else:
            self.weights = {"dq": np.random.normal(0, 0.05, (input_dim, 3)).astype(np.float32)}

    def forward(self, x: Any) -> Any:
        if TORCH_AVAILABLE and isinstance(self.net, nn.Module):
            if x.dim() == 3: x = torch.mean(x, dim=1)
            elif x.dim() == 1: x = x.unsqueeze(0)
            return self.net(x)
        return np.array([0.1, 0.4, -0.1], dtype=np.float32)


class RecurrentQLearningModel:
    """Recurrent Q-Learning Agent (`agent/recurrent-q-learning-agent`)."""
    def __init__(self, input_dim: int = 8, hidden_dim: int = 16):
        self.model_name = "Recurrent-Q-learning-Agent"
        self.input_dim = input_dim
        if TORCH_AVAILABLE:
            self.rnn = nn.GRU(input_dim, hidden_dim, batch_first=True)
            self.fc = nn.Linear(hidden_dim, 3)
            self.net = nn.ModuleList([self.rnn, self.fc])
        else:
            self.weights = {"rq": np.random.normal(0, 0.05, (input_dim, 3)).astype(np.float32)}

    def forward(self, x: Any) -> Any:
        if TORCH_AVAILABLE and hasattr(self, "rnn"):
            if x.dim() == 2: x = x.unsqueeze(0)
            _, h_n = self.rnn(x)
            return self.fc(h_n[-1])
        return np.array([0.1, 0.5, -0.2], dtype=np.float32)


class CuriosityQLearningModel:
    """Intrinsic Curiosity Q-Learning Agent with Forward Dynamics Model (`agent/curiosity-q-learning-agent`)."""
    def __init__(self, input_dim: int = 8, hidden_dim: int = 16):
        self.model_name = "Curiosity-Q-learning-Agent"
        self.input_dim = input_dim
        if TORCH_AVAILABLE:
            self.q_net = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 3))
            # Forward Dynamics Model: predicts next state from current state + action
            self.dynamics_net = nn.Sequential(nn.Linear(input_dim + 1, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, input_dim))
            self.net = nn.ModuleList([self.q_net, self.dynamics_net])
        else:
            self.weights = {"q": np.random.normal(0, 0.05, (input_dim, 3)).astype(np.float32)}

    def forward(self, x: Any) -> Any:
        if TORCH_AVAILABLE and hasattr(self, "q_net"):
            if x.dim() == 3: x = torch.mean(x, dim=1)
            elif x.dim() == 1: x = x.unsqueeze(0)
            return self.q_net(x)
        return np.array([0.1, 0.6, -0.1], dtype=np.float32)

    def compute_curiosity_bonus(self, state: np.ndarray, action: float, next_state: np.ndarray) -> float:
        """Computes prediction error ||s_{t+1} - f(s_t, a_t)||^2 as intrinsic novelty bonus."""
        if TORCH_AVAILABLE and hasattr(self, "dynamics_net"):
            s_tensor = torch.from_numpy(state).to(torch.float32)
            if s_tensor.dim() == 2: s_tensor = torch.mean(s_tensor, dim=0)
            a_tensor = torch.tensor([action], dtype=torch.float32)
            inp = torch.cat([s_tensor, a_tensor], dim=0).unsqueeze(0)
            pred_next = self.dynamics_net(inp)
            true_next = torch.from_numpy(next_state).to(torch.float32)
            if true_next.dim() == 2: true_next = torch.mean(true_next, dim=0)
            bonus = float(torch.mean((pred_next.squeeze(0) - true_next) ** 2).item())
            return min(2.0, bonus)
        return 0.05


class NeuroEvolutionNESModel:
    """Natural Evolution Strategies (NES) Parameter Agent (`agent/neuro-evolution-agent`)."""
    def __init__(self, input_dim: int = 8, hidden_dim: int = 16):
        self.model_name = "Neuro-Evolution-Agent"
        self.input_dim = input_dim
        self.weights = {
            "w1": np.random.normal(0, 0.1, (input_dim, hidden_dim)).astype(np.float32),
            "w2": np.random.normal(0, 0.1, (hidden_dim, 1)).astype(np.float32)
        }
        self.population_size = 10
        self.sigma = 0.1
        self.learning_rate = 0.02

    def forward(self, x: Any) -> Any:
        if TORCH_AVAILABLE and isinstance(x, torch.Tensor):
            x_arr = x.cpu().numpy()
        else:
            x_arr = np.array(x, dtype=np.float32)
        if x_arr.ndim == 3: x_arr = np.mean(x_arr, axis=1)
        elif x_arr.ndim == 1: x_arr = np.expand_dims(x_arr, axis=0)
        h = np.maximum(0, np.dot(x_arr, self.weights["w1"]))
        out = np.dot(h, self.weights["w2"])
        return float(out[0, 0]) if out.size > 0 else 0.0

    def evolve_step(self, fitness_scores: List[float], noise_samples: List[Dict[str, np.ndarray]]) -> None:
        """Performs NES gradient-free parameter update given evaluated population fitness."""
        if not fitness_scores or not noise_samples: return
        std_fit = np.std(fitness_scores)
        if std_fit == 0: return
        norm_fitness = (np.array(fitness_scores) - np.mean(fitness_scores)) / std_fit
        for k in self.weights:
            step = np.zeros_like(self.weights[k])
            for idx, fit in enumerate(norm_fitness):
                step += fit * noise_samples[idx][k]
            self.weights[k] += self.learning_rate / (self.population_size * self.sigma) * step


class NeuroEvolutionNoveltySearchModel:
    """Neuro-Evolution Novelty Search Agent (`agent/neuro-evolution-novelty-search-agent`)."""
    def __init__(self, input_dim: int = 8, hidden_dim: int = 16):
        self.model_name = "Neuro-Evolution-Novelty-Search-Agent"
        self.input_dim = input_dim
        if TORCH_AVAILABLE:
            self.net = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1))
        else:
            self.weights = {"ns": np.random.normal(0, 0.1, (input_dim, 1)).astype(np.float32)}
        self.behavior_archive: List[float] = []

    def forward(self, x: Any) -> Any:
        if TORCH_AVAILABLE and isinstance(self.net, nn.Module):
            if isinstance(x, np.ndarray): x = torch.from_numpy(x).to(torch.float32)
            if x.dim() == 3: x = torch.mean(x, dim=1)
            elif x.dim() == 1: x = x.unsqueeze(0)
            out = self.net(x)
            val = float(out.mean().item())
            if len(self.behavior_archive) < 100: self.behavior_archive.append(val)
            return out
        return np.sum(x, axis=-1, keepdims=True) * 0.01

    def compute_novelty_distance(self, behavior_point: float) -> float:
        """Computes average distance to k-nearest neighbors in behavioral archive."""
        if not self.behavior_archive: return 1.0
        dists = sorted(abs(behavior_point - b) for b in self.behavior_archive)
        k = min(5, len(dists))
        return sum(dists[:k]) / k if k > 0 else 1.0


class MovingAverageMomentumModel:
    """Algorithmic Moving Average & Statistical Momentum Scalper (`agent/moving-average-agent` / `abcd-strategy-agent`)."""
    def __init__(self, input_dim: int = 8, short_window: int = 5, long_window: int = 20):
        self.model_name = "Moving-Average-Agent"
        self.short_window = short_window
        self.long_window = long_window

    def forward(self, x: Any) -> Any:
        if TORCH_AVAILABLE and isinstance(x, torch.Tensor):
            x_arr = x.cpu().numpy()
        else:
            x_arr = np.array(x, dtype=np.float32)
        if x_arr.ndim == 3: prices = x_arr[0, :, 0]
        elif x_arr.ndim == 2: prices = x_arr[:, 0]
        else: prices = x_arr
        if len(prices) < self.long_window: return 0.0
        short_ma = np.mean(prices[-self.short_window:])
        long_ma = np.mean(prices[-self.long_window:])
        diff = (short_ma - long_ma) / (long_ma + 1e-8)
        return float(np.clip(diff * 50.0, -1.0, 1.0))


# ─── UNIVERSAL REGISTRY MANAGER (`TradeJackModelRegistry`) ───

class TradeJackModelRegistry:
    """
    Central Policy Card Registry linking all model builders to their VRAM/survival tier boundaries.
    """
    def __init__(self):
        self.cards: Dict[str, ModelCard] = {}
        self._register_all_models()

    def _register(self, name: str, tier: int, cat: str, vram_mb: float, desc: str, builder: Callable[..., Any]):
        self.cards[name] = ModelCard(
            model_name=name,
            tier_requirement=tier,
            category=cat,
            vram_estimate_mb=vram_mb,
            description=desc,
            builder_fn=builder
        )

    def _register_all_models(self):
        # Tier 1 (<= 20GB VRAM / High Alpha)
        self._register("Attention-is-all-you-Need", 1, "transformer", 2500.0, "Positional Causal Transformer", AttentionIsAllYouNeedModel)
        self._register("LSTM-Seq2Seq-VAE", 1, "seq2seq_vae", 1800.0, "LSTM Variational Autoencoder", LSTMSeq2SeqVAEModel)
        self._register("GRU-Seq2Seq-VAE", 1, "seq2seq_vae", 1600.0, "GRU Variational Autoencoder", GRUSeq2SeqVAEModel)
        self._register("Stacking-Encoder-Ensemble", 1, "ensemble", 2200.0, "Autoencoder + Multi-Path Stacking", StackingEncoderEnsembleModel)
        self._register("Stack-RNN-ARIMA-XGB", 1, "ensemble", 1900.0, "Hybrid Recurrent + Statistical Ensemble", StackRNNARIMAXGBModel)
        
        # Tier 2 (<= 4GB VRAM / Standard Sequence & Actor-Critic)
        self._register("Dilated-CNN-Seq2seq", 2, "cnn", 850.0, "Stacked Residual Causal Dilated CNN", DilatedCNNSeq2SeqModel)
        self._register("CNN-Seq2seq", 2, "cnn", 600.0, "Causal 1D Convolutional Network", CNNSeq2SeqModel)
        self._register("LSTM-Seq2seq", 2, "rnn", 750.0, "Standard LSTM Sequence Model", LSTMSeq2SeqModel)
        self._register("Bidirectional-LSTM-Seq2seq", 2, "rnn", 950.0, "Bidirectional LSTM Sequence Model", BiLSTMSeq2SeqModel)
        self._register("GRU-Seq2seq", 2, "rnn", 650.0, "Standard GRU Sequence Model", GRUSeq2SeqModel)
        self._register("Vanilla-Seq2seq", 2, "rnn", 400.0, "Vanilla RNN Sequence Model", VanillaSeq2SeqModel)
        self._register("Actor-Critic-Agent", 2, "actor_critic", 700.0, "A2C/PPO Policy & Value Network", ActorCriticAgentModel)
        self._register("Actor-Critic-Duel-Agent", 2, "actor_critic", 780.0, "Dueling Advantage Actor-Critic", ActorCriticDuelAgentModel)
        self._register("Actor-Critic-Recurrent-Agent", 2, "actor_critic", 850.0, "Recurrent LSTM Actor-Critic", ActorCriticRecurrentAgentModel)
        self._register("Double-Duel-Recurrent-Q-Learning-Agent", 2, "q_learning", 820.0, "Double Dueling Recurrent Q-Learning", DoubleDuelRecurrentQAgentModel)
        
        # Tier 3 (<= 1GB VRAM / Critical Survival & Scalpers)
        self._register("Deep-Q-learning", 3, "q_learning", 220.0, "Dueling Q-Network Micro-Scalper", DeepQLearningModel)
        self._register("Double-Q-learning-Agent", 3, "q_learning", 180.0, "Double Q-Learning Scalper", DoubleQLearningModel)
        self._register("Recurrent-Q-learning-Agent", 3, "q_learning", 260.0, "Recurrent GRU Q-Learning Scalper", RecurrentQLearningModel)
        self._register("Curiosity-Q-learning-Agent", 3, "q_learning", 310.0, "Intrinsic Curiosity Reward Scalper", CuriosityQLearningModel)
        self._register("Neuro-Evolution-Agent", 3, "neuro_evolution", 80.0, "Natural Evolution Strategies (NES)", NeuroEvolutionNESModel)
        self._register("Neuro-Evolution-Novelty-Search-Agent", 3, "neuro_evolution", 120.0, "Novelty Behavioral Search Agent", NeuroEvolutionNoveltySearchModel)
        self._register("Moving-Average-Agent", 3, "q_learning", 20.0, "Algorithmic Moving Average Momentum Scalper", MovingAverageMomentumModel)

    def get_model_card(self, model_name: str) -> Optional[ModelCard]:
        return self.cards.get(model_name)

    def list_models_for_tier(self, max_tier: int) -> List[ModelCard]:
        """Returns all model cards whose tier requirement is greater than or equal to `max_tier` (where Tier 3 is lowest compute, Tier 1 is highest)."""
        return [card for card in self.cards.values() if card.tier_requirement >= max_tier]

    def build_model(self, model_name: str, input_dim: int = 8, **kwargs) -> Any:
        """Instantiates the live model architecture by name."""
        card = self.get_model_card(model_name)
        if not card:
            logger.warning(f"Model '{model_name}' not found in registry. Defaulting to 'Deep-Q-learning'.")
            card = self.cards["Deep-Q-learning"]
        return card.builder_fn(input_dim=input_dim, **kwargs)


# Global singleton instance
REGISTRY = TradeJackModelRegistry()
