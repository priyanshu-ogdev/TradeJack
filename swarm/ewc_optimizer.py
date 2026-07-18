"""
Elastic Weight Consolidation (EWC) Optimizer Wrapper (`ewc_optimizer.py`).
Computes diagonal Fisher Information Matrix across proven historical survival trajectories.
Adds EWC penalty (`lambda / 2 * sum(F_i * (theta - theta_old)^2)`) to task loss during `self-mod/` fine-tuning,
preventing Catastrophic Forgetting of past alpha and regime adaptations.
"""

import os
import sys
import logging
import numpy as np
from typing import Dict, Any, List, Optional, Callable

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] (EWCOptimizer) %(message)s")
logger = logging.getLogger("EWCOptimizer")

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.info("PyTorch not installed; EWCOptimizer operating in Numpy simulation fallback mode.")


class ElasticWeightConsolidation:
    """
    Computes Fisher Information and enforces EWC penalty across PyTorch or Numpy model weights.
    """

    def __init__(
        self,
        model: Any,
        data_loader: Any,
        ewc_lambda: float = 1000.0,
        device: str = "cuda:0" if (TORCH_AVAILABLE and torch.cuda.is_available()) else "cpu"
    ):
        self.model = model
        self.ewc_lambda = ewc_lambda
        self.device = device
        self.star_params: Dict[str, Any] = {}
        self.fisher_matrix: Dict[str, Any] = {}
        
        if TORCH_AVAILABLE and isinstance(model, nn.Module):
            self.model.to(self.device)
            self._compute_fisher_pytorch(data_loader)
        else:
            self._compute_fisher_numpy(data_loader)

    def _compute_fisher_pytorch(self, data_loader: Any):
        """Computes diagonal Fisher Information Matrix across calibration dataloader in PyTorch."""
        self.star_params = {n: p.clone().detach() for n, p in self.model.named_parameters() if p.requires_grad}
        self.fisher_matrix = {n: torch.zeros_like(p) for n, p in self.model.named_parameters() if p.requires_grad}
        
        self.model.eval()
        count = 0
        
        if data_loader is not None:
            for batch_x, batch_y in data_loader:
                self.model.zero_grad()
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                outputs = self.model(batch_x)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                    
                # Compute log likelihood / loss gradients
                loss = F.mse_loss(outputs.squeeze(), batch_y.squeeze())
                loss.backward()
                
                for n, p in self.model.named_parameters():
                    if p.requires_grad and p.grad is not None:
                        self.fisher_matrix[n] += p.grad.detach() ** 2
                count += 1
                if count >= 20:  # Calibrate over top 20 batches
                    break
                    
        if count > 0:
            for n in self.fisher_matrix:
                self.fisher_matrix[n] /= float(count)
        logger.info(f"PyTorch EWC Fisher Information computed across {count} batches.")

    def _compute_fisher_numpy(self, data_loader: Any):
        """Numpy simulation fallback for Fisher calculation when PyTorch is not installed."""
        if hasattr(self.model, "weights") and isinstance(self.model.weights, dict):
            self.star_params = {k: np.copy(v) for k, v in self.model.weights.items()}
            self.fisher_matrix = {k: np.ones_like(v, dtype=np.float32) * 0.1 for k, v in self.model.weights.items()}
        logger.info("Numpy EWC Fisher Information matrix initialized.")

    def penalty(self, model_instance: Optional[Any] = None) -> Any:
        """
        Computes the EWC regularization penalty: lambda / 2 * sum(F_i * (theta_i - theta_star_i)^2).
        Must be added to task loss during backward pass.
        """
        target_model = model_instance or self.model
        
        if TORCH_AVAILABLE and isinstance(target_model, nn.Module):
            loss = torch.tensor(0.0, device=self.device)
            for n, p in target_model.named_parameters():
                if n in self.fisher_matrix and n in self.star_params:
                    loss += (self.fisher_matrix[n] * (p - self.star_params[n]) ** 2).sum()
            return loss * (self.ewc_lambda / 2.0)
        else:
            # Numpy penalty
            loss = 0.0
            if hasattr(target_model, "weights") and isinstance(target_model.weights, dict):
                for k, v in target_model.weights.items():
                    if k in self.fisher_matrix and k in self.star_params:
                        loss += float(np.sum(self.fisher_matrix[k] * (v - self.star_params[k]) ** 2))
            return loss * (self.ewc_lambda / 2.0)


if __name__ == "__main__":
    logger.info("Testing ElasticWeightConsolidation standalone...")
    # Test numpy fallback object
    class DummyModel:
        def __init__(self):
            self.weights = {"layer1": np.ones((5, 5), dtype=np.float32)}
    model = DummyModel()
    ewc = ElasticWeightConsolidation(model, None)
    # Simulate weight drift
    model.weights["layer1"] += 0.5
    print("Computed EWC penalty after weight drift:", ewc.penalty(model))
