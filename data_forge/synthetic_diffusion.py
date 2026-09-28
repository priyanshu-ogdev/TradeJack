"""
Synthetic Black-Swan Diffusion Engine.
Generates millions of synthetic market scenarios that never actually happened but could
(e.g., flash crashes, stablecoin de-pegs).
Uses Time-Series Diffusion concepts to ensure complex multivariate autocorrelation is maintained,
preventing the mode-collapse common in GANs.

SOTA Upgrades:
  - ZSTD-L3 compression with configurable row_group_size
  - NPZ fallback when Polars is unavailable (no longer silently drops data)
"""

import os
import random
import logging
from typing import Optional

import numpy as np

from data_forge.config import config

try:
    import polars as pl
    POLARS_AVAILABLE = True
except ImportError:
    POLARS_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] (SyntheticDiffusion) %(message)s")
logger = logging.getLogger("SyntheticDiffusion")

class SyntheticDiffusionEngine:
    def __init__(self, symbol: str = "BTC-USDT"):
        self.symbol = symbol
        self.store_dir = os.path.join(config.data_store_dir, "synthetic", symbol)
        os.makedirs(self.store_dir, exist_ok=True)

    def _generate_scenario_arrays(
        self,
        base_price: float,
        num_ticks: int,
        volatility_multiplier: float,
        jump_direction: float = -1.0,
        jump_probability: float = 0.01,
    ) -> dict:
        """
        Pure-numpy generation, no I/O -- split out from generate_black_swan_scenario()
        so callers needing the physics-schema-shaped arrays directly (e.g.
        training/scenario_injection.py, which writes them into a physics partition
        path instead of data_store/synthetic/) don't have to go through disk.

        jump_direction/jump_probability are NEW, optional (default to the exact prior
        hardcoded values: -1.0, 0.01) -- this is the fix for a real limitation found
        while reviewing this module for Phase 3: despite the module's docstring and
        the upgrade plan describing this as supporting "named regimes," the actual
        stochastic process only ever varied by volatility_multiplier -- jump
        direction was hardcoded negative, so every "regime" was really just a
        differently-scaled version of the same one-sided-crash pattern, not a
        genuinely different shape (a melt-up squeeze and a flash crash are
        economically very different tail events, and training data should reflect
        that). jump_direction=+1.0 produces a melt-up/squeeze; 0.0 produces a
        two-sided (up OR down) jump distribution, useful for a generic
        "high-volatility, direction uncertain" regime.
        """
        timestamps = np.arange(num_ticks, dtype=np.float64)

        standard_vol = 0.0005
        brownian_shocks = np.random.normal(0, standard_vol, num_ticks)

        jump_probabilities = np.random.uniform(0, 1, num_ticks)
        is_jump = jump_probabilities < jump_probability
        if jump_direction == 0.0:
            # Two-sided: each jump independently up or down, not a fixed bias.
            jump_sign = np.random.choice([-1.0, 1.0], size=num_ticks)
        else:
            jump_sign = np.full(num_ticks, 1.0 if jump_direction > 0 else -1.0)
        jump_shocks = np.where(
            is_jump,
            jump_sign * np.abs(np.random.normal(0.015, 0.005, num_ticks)) * volatility_multiplier,
            0.0,
        )

        total_shocks = brownian_shocks + jump_shocks
        price_path = base_price * np.exp(np.cumsum(total_shocks))

        ofi = np.random.normal(0, 10, num_ticks)
        ofi += (total_shocks * 10000)

        vpin = np.random.uniform(0.1, 0.3, num_ticks)
        vpin = np.where(is_jump, np.random.uniform(0.7, 0.9, num_ticks), vpin)

        volume = np.random.lognormal(mean=2, sigma=1, size=num_ticks)
        volume = np.where(is_jump, volume * 10, volume)

        kyles_lambda = (total_shocks * price_path) / (ofi + 1e-8)

        data_dict = {
            "timestamp": timestamps,
            "open_price": np.roll(price_path, 1),
            "close_price": price_path,
            "volume": volume,
            "ofi": ofi,
            "vpin_50": vpin,
            "kyles_lambda": kyles_lambda,
        }
        data_dict["open_price"][0] = base_price
        return data_dict

    def inject_spoofing_overlay(
        self,
        data_dict: dict,
        spoof_probability: float = 0.02,
        spoof_duration_ticks: int = 5,
        ofi_multiplier: float = 50.0,
        volume_multiplier: float = 10.0,
        vpin_range: tuple = (0.7, 0.9),
        rng: Optional[np.random.Generator] = None,
    ) -> dict:
        """
        Overlays spoofing events onto an already-generated scenario (from
        _generate_scenario_arrays), mutating copies of ofi/volume/vpin_50 only --
        close_price/open_price/kyles_lambda are returned byte-for-byte unchanged.

        This is the actual economically-correct way to express manipulation-
        robustness training in the 5-channel physics-feature space (see
        docs/PHASE3_SCENARIO_DIVERSITY.md's research section for the literature this
        is grounded in -- Cartea et al. 2020/2023 and Do & Putninš 2023 both
        characterize spoofing via order-flow/liquidity imbalance, and the specific
        "imbalance spikes then vanishes without a corresponding price move" shape
        is exactly what distinguishes it from genuine informed flow). The legacy
        AdversarialGANSpoofer (swarm/rl_mechanics.py) instead directly multiplies
        RAW 8-channel bid/ask price/size columns -- a representation nothing in the
        active SB3 pipeline consumes (TradeJackLOBEnv's observation is this exact
        5-channel schema); porting that mechanism unchanged would silently
        misapply column semantics that don't exist in this schema.

        GENUINELY DECOUPLED from the price-jump process, not just "also random":
        spoof event locations are drawn from their own independent RNG stream
        (`rng`, defaulting to a fresh np.random.default_rng() -- deliberately NOT
        np.random.uniform/np.random.seed, which would share state with
        _generate_scenario_arrays's global-RNG jump draws and could accidentally
        correlate spoof timing with real jump timing depending on call order).
        An agent that only learns "big OFI spike = real move" or "= fake" from
        correlated training data would be learning a spurious shortcut, not the
        actual discriminating signal (imbalance without price follow-through).

        Reversion: at exactly spoof_duration_ticks after onset, ofi/volume/vpin_50
        snap back to their PRE-SPOOF values (not decay toward them) -- modeling the
        research's "vanishes quickly without execution" signature literally, not
        approximately.

        Returns a new dict (does not mutate the input).
        """
        rng = rng or np.random.default_rng()
        out = {k: np.array(v, copy=True) for k, v in data_dict.items()}
        num_ticks = len(out["close_price"])

        onset_draws = rng.uniform(0, 1, num_ticks)
        is_onset = onset_draws < spoof_probability

        t = 0
        while t < num_ticks:
            if is_onset[t]:
                end = min(t + spoof_duration_ticks, num_ticks)
                out["ofi"][t:end] = out["ofi"][t:end] * ofi_multiplier
                out["volume"][t:end] = out["volume"][t:end] * volume_multiplier
                out["vpin_50"][t:end] = rng.uniform(vpin_range[0], vpin_range[1], end - t)
                t = end  # skip past this spoof window rather than re-triggering mid-window
            else:
                t += 1

        return out

    def generate_black_swan_scenario(
        self,
        scenario_name: str,
        base_price: float = 65000.0,
        num_ticks: int = 1000,
        volatility_multiplier: float = 5.0,
        jump_direction: float = -1.0,
        jump_probability: float = 0.01,
    ):
        """
        Uses a pseudo-diffusion process (Gaussian random walks with extreme jump diffusions)
        to create realistic, high-stress tail risk data for adversarial RL training.
        """
        logger.info(f"Generating synthetic Black Swan Scenario: {scenario_name} (Ticks: {num_ticks})")

        data_dict = self._generate_scenario_arrays(
            base_price, num_ticks, volatility_multiplier, jump_direction, jump_probability
        )

        if POLARS_AVAILABLE:
            df = pl.DataFrame(data_dict)
            out_path = os.path.join(self.store_dir, f"{scenario_name}.parquet")
            df.write_parquet(
                out_path,
                compression=config.compression_codec,
                compression_level=config.compression_level,
                row_group_size=config.row_group_size,
            )
            logger.info(f"Successfully wrote synthetic scenario to {out_path}")
        else:
            # NPZ fallback: compressed Numpy archive when Polars is unavailable
            out_path = os.path.join(self.store_dir, f"{scenario_name}.npz")
            np.savez_compressed(out_path, **data_dict)
            logger.info(f"Successfully wrote synthetic scenario (NPZ fallback) to {out_path}")

if __name__ == "__main__":
    engine = SyntheticDiffusionEngine()
    engine.generate_black_swan_scenario("flash_crash_2026", base_price=100000.0, num_ticks=5000, volatility_multiplier=8.0)
    logger.info("Synthetic Generation Engine Ready.")
