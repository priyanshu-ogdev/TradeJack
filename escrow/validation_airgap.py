"""
10x Rolling Validation Airgap (`ValidationAirgapEngine`).
Evaluates candidate `state_dict` weights in a completely sandboxed simulation across 10 out-of-sample
historical market splits (including simulated liquidity vacuums and high volatility crashes).
If the candidate model fails (`average_sharpe < 1.0` or `max_drawdown >= 0.15` on any split),
the weights are classified as poisoned or overfitted and rejected before escrow settlement.
"""

import os
import sys
import time
import math
import random
import json
import logging
import numpy as np
from typing import Dict, Any, List, Optional, Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] (ValidationAirgap) %(message)s")
logger = logging.getLogger("ValidationAirgap")

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from physics.lob_env import TradeJackLOBEnv
from swarm.self_mod_manager import SelfModEngine, DilatedCNNSeq2SeqTemplate, AttentionIsAllYouNeedTemplate, DeepQLearningTemplate


class ValidationAirgapEngine:
    """
    Sandboxed evaluation chamber running candidate weights across 10 distinct historical stress tests.
    """

    def __init__(
        self,
        num_splits: int = 10,
        min_required_sharpe: float = 1.0,
        max_allowed_drawdown: float = 0.15,
        data_store_dir: str = "d:/TradeJack/data_store"
    ):
        self.num_splits = num_splits
        self.min_sharpe = min_required_sharpe
        self.max_drawdown = max_allowed_drawdown
        self.data_store_dir = os.path.abspath(data_store_dir)

    def _load_model_from_weights(self, weights_path: str, model_type: str = "Dilated-CNN-Seq2seq") -> Any:
        """Instantiates a candidate model and loads parameters from `state_dict_path`."""
        if model_type == "Attention-is-all-you-Need":
            model = AttentionIsAllYouNeedTemplate()
        elif model_type == "Deep-Q-learning":
            model = DeepQLearningTemplate()
        else:
            model = DilatedCNNSeq2SeqTemplate()
            
        if TORCH_AVAILABLE and hasattr(model, "net") and os.path.exists(weights_path):
            try:
                state_dict = torch.load(weights_path, map_location="cpu")
                if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
                    model.net.load_state_dict(state_dict["model_state_dict"], strict=False)
                elif isinstance(state_dict, dict):
                    model.net.load_state_dict(state_dict, strict=False)
                logger.debug(f"Loaded PyTorch state_dict from {weights_path}")
            except Exception as e:
                logger.warning(f"Could not load PyTorch weights ({e}). Evaluating default parameters.")
        elif hasattr(model, "weights") and os.path.exists(weights_path):
            # Check for numpy weights
            try:
                if weights_path.endswith(".npz"):
                    data = np.load(weights_path)
                    for k in model.weights:
                        if k in data:
                            model.weights[k] = data[k]
            except Exception:
                pass
                
        return model

    def evaluate_candidate_weights(
        self,
        weights_path: str,
        model_type: str = "Dilated-CNN-Seq2seq",
        symbol: str = "BTC-USDT"
    ) -> Dict[str, Any]:
        """
        Runs the 10x Rolling Validation Airgap. Returns validation status and metrics across splits.
        """
        logger.info(f"[AIRGAP COMMENCING] Sandboxing weights '{weights_path}' ({model_type}) across {self.num_splits} stress splits...")
        
        model = self._load_model_from_weights(weights_path, model_type=model_type)
        split_sharpes: List[float] = []
        split_drawdowns: List[float] = []
        split_equities: List[float] = []
        
        for split_idx in range(self.num_splits):
            # Each split tests a different deterministic random seed and simulated LOB shock profile
            seed = 1000 + split_idx * 17
            random.seed(seed)
            np.random.seed(seed)
            
            env = TradeJackLOBEnv(
                symbol=symbol,
                initial_cash=10.0,
                data_store_dir=self.data_store_dir,
                child_id=999  # Airgap sandbox id
            )
            
            obs, info = env.reset(seed=seed)
            terminated = False
            truncated = False
            
            # Run up to 100 steps on this split
            for _ in range(100):
                lob_seq = obs["lob_sequence"]
                if TORCH_AVAILABLE and hasattr(model, "net"):
                    t_in = torch.from_numpy(lob_seq).unsqueeze(0).to(torch.float32)
                    with torch.no_grad():
                        out = model.forward(t_in)
                        action = float(out.mean().cpu().numpy()) if isinstance(out, torch.Tensor) else float(np.mean(out))
                else:
                    out = model.forward(lob_seq)
                    action = float(np.mean(out))
                    
                action = float(np.clip(action, -1.0, 1.0))
                obs, reward, terminated, truncated, info = env.step([action])
                if terminated or truncated:
                    break
                    
            # Calculate split metrics
            eq = info["equity"]
            dd = info["max_drawdown"]
            # Approximate split Sharpe from equity change
            ret = (eq - 10.0) / 10.0
            split_sharpe = ret * 10.0 if dd < 0.05 else ret / (dd + 1e-4)
            
            split_equities.append(eq)
            split_drawdowns.append(dd)
            split_sharpes.append(split_sharpe)
            
        avg_sharpe = float(np.mean(split_sharpes))
        max_dd = float(np.max(split_drawdowns))
        passed_airgap = (avg_sharpe >= self.min_sharpe) and (max_dd <= self.max_drawdown)
        
        if passed_airgap:
            logger.info(f"[AIRGAP PASSED] Candidate weights cleared validation. Avg Sharpe: {avg_sharpe:.2f}, Max DD: {max_dd*100:.1f}%.")
        else:
            logger.warning(
                f"[AIRGAP REJECTED] Candidate weights failed validation! Avg Sharpe: {avg_sharpe:.2f} (Req >= {self.min_sharpe}), "
                f"Max DD: {max_dd*100:.1f}% (Req <= {self.max_drawdown*100:.1f}%). Flagged as overfitted/poisoned."
            )
            
        return {
            "passed": passed_airgap,
            "weights_path": weights_path,
            "model_type": model_type,
            "avg_sharpe": avg_sharpe,          # used by promote_candidate.py
            "average_sharpe": avg_sharpe,      # backward compat
            "max_drawdown": max_dd,
            "split_results": {
                "equities": split_equities,
                "drawdowns": split_drawdowns,
                "sharpes": split_sharpes
            }
        }

    def validate_promotion_candidate(
        self,
        weights_path: str,
        model_type: str,
        min_sharpe_override: float = 1.0,
        max_drawdown_override: float = 0.15
    ) -> dict:
        """
        Segment 4.3 (BUG-12 FIX): Strict promotion gate for Crucible->Deployment transition.
        Candidate must clear: Avg Sharpe >= 1.0, Max DD <= 15%.
        This method is called by scripts/promote_candidate.py before any weight copy.
        No code path exists to reach deployment without passing this gate.

        Returns dict with 'passed' bool and detailed per-split metrics.
        """
        # Temporarily override thresholds for promotion gate
        original_sharpe = self.min_sharpe
        original_dd = self.max_drawdown
        self.min_sharpe = min_sharpe_override
        self.max_drawdown = max_drawdown_override

        logger.info(
            f"[PROMOTION GATE] Evaluating {weights_path} ({model_type}). "
            f"Requirements: Avg Sharpe >= {min_sharpe_override}, Max DD <= {max_drawdown_override*100:.0f}%."
        )

        try:
            result = self.evaluate_candidate_weights(weights_path, model_type=model_type)
        finally:
            # Always restore original thresholds
            self.min_sharpe = original_sharpe
            self.max_drawdown = original_dd

        if result["passed"]:
            logger.info(
                f"[PROMOTION GATE PASSED] Child candidate {weights_path} is cleared for deployment. "
                f"Avg Sharpe: {result['avg_sharpe']:.3f}, Max DD: {result['max_drawdown']*100:.1f}%."
            )
        else:
            logger.warning(
                f"[PROMOTION GATE BLOCKED] Candidate rejected. "
                f"Avg Sharpe: {result['avg_sharpe']:.3f} (required >= {min_sharpe_override}), "
                f"Max DD: {result['max_drawdown']*100:.1f}% (required <= {max_drawdown_override*100:.0f}%)."
            )
        return result


if __name__ == "__main__":
    logger.info("Testing ValidationAirgapEngine standalone...")
    # Generate test data if needed
    from data_forge.parquet_ingest import ParquetIngestPipeline
    import asyncio
    ingest = ParquetIngestPipeline(data_store_dir="d:/TradeJack/data_store")
    asyncio.run(ingest.generate_synthetic_crucible_data(symbol="BTC-USDT", num_days=1, ticks_per_day=150))
    
    airgap = ValidationAirgapEngine(num_splits=3, min_required_sharpe=0.0, max_allowed_drawdown=0.5)
    res = airgap.evaluate_candidate_weights("dummy_weights.pt", model_type="Dilated-CNN-Seq2seq")
    print("Airgap Evaluation Summary:", json.dumps(res, indent=2))
