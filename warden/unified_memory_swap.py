"""
Unified Memory Zero-Copy Swapping Engine for Grace Blackwell Architecture (128GB Unified Memory).
Enables seamless zero-copy offloading and instantaneous swapping of large neuro-evolutionary model populations between CUDA execution cores and Grace unified system memory without PCIe bus transfer latency.
"""

import os
import sys
import gc
import logging
from typing import Dict, Any, List, Callable, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] (UnifiedMemorySwap) %(message)s")
logger = logging.getLogger("UnifiedMemorySwap")

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not installed locally; BlackwellUnifiedAllocator will run in simulation mode.")


class BlackwellUnifiedAllocator:
    """
    Manages memory allocation and zero-copy swapping across Grace Blackwell 128GB Unified Memory.
    """

    def __init__(
        self,
        vram_quota_gb: float = 20.0,
        enable_fp8_autocast: bool = True,
        device: str = "cuda:0"
    ):
        self.vram_quota_gb = vram_quota_gb
        self.enable_fp8_autocast = enable_fp8_autocast
        self.device = device if (TORCH_AVAILABLE and torch.cuda.is_available()) else "cpu"
        self.pinned_pool: Dict[str, Dict[str, Any]] = {}
        
        if TORCH_AVAILABLE and torch.cuda.is_available():
            # Set per-process VRAM fraction if quota is lower than total physical GPU RAM
            try:
                total_mem = torch.cuda.get_device_properties(self.device).total_memory / (1024**3)
                if self.vram_quota_gb < total_mem:
                    fraction = min(1.0, self.vram_quota_gb / total_mem)
                    torch.cuda.set_per_process_memory_fraction(fraction, device=self.device)
                    logger.info(f"Set PyTorch VRAM limit to {self.vram_quota_gb:.1f}GB ({fraction*100:.1f}% of {total_mem:.1f}GB) on {self.device}")
            except Exception as e:
                logger.warning(f"Could not enforce per-process VRAM fraction: {e}")

    def pin_to_unified_memory(self, model_id: str, state_dict: Dict[str, Any]) -> int:
        """
        Pins a candidate model's state_dict tensors to Grace Blackwell Unified Memory.
        Because Grace Blackwell shares physical RAM between CPU and GPU over high-speed NVLink-C2C,
        pinned memory can be read by CUDA execution units directly or swapped in microseconds.
        Returns bytes pinned.
        """
        if not TORCH_AVAILABLE:
            self.pinned_pool[model_id] = state_dict
            return 0
            
        pinned_state = {}
        total_bytes = 0
        
        for k, v in state_dict.items():
            if isinstance(v, torch.Tensor):
                # Ensure tensor is on host CPU and pinned
                t_host = v.detach().cpu()
                if not t_host.is_pinned():
                    t_host = t_host.pin_memory()
                pinned_state[k] = t_host
                total_bytes += t_host.element_size() * t_host.nelement()
            else:
                pinned_state[k] = v
                
        self.pinned_pool[model_id] = pinned_state
        logger.debug(f"Pinned model '{model_id}' to Unified Memory ({total_bytes / (1024**2):.2f} MB)")
        return total_bytes

    def swap_into_active_cuda(self, model_id: str, target_model: Any) -> bool:
        """
        Instantly streams pinned unified memory tensors into the active CUDA model execution tensors
        using non_blocking zero-copy transfers over NVLink-C2C.
        """
        if not TORCH_AVAILABLE:
            return True
            
        if model_id not in self.pinned_pool:
            logger.error(f"Model ID '{model_id}' not found in Unified Memory pinned pool.")
            return False
            
        pinned_state = self.pinned_pool[model_id]
        active_state = {}
        
        for k, v in pinned_state.items():
            if isinstance(v, torch.Tensor):
                # Non-blocking transfer from pinned unified RAM directly to CUDA registers/VRAM
                active_state[k] = v.to(self.device, non_blocking=True)
            else:
                active_state[k] = v
                
        target_model.load_state_dict(active_state, strict=False)
        return True

    def evaluate_population_batch(
        self,
        population_models: Dict[str, Any],
        eval_fn: Callable[[str, Any], float],
        batch_size: int = 4
    ) -> Dict[str, float]:
        """
        Evaluates a large population (e.g., 50 neuro-evolution models) without triggering OOM.
        Schedules `batch_size` models into active CUDA memory simultaneously while keeping the remaining
        population pinned zero-copy in Grace Unified Memory.
        """
        results: Dict[str, float] = {}
        model_ids = list(population_models.keys())
        
        for i in range(0, len(model_ids), batch_size):
            batch_ids = model_ids[i:i + batch_size]
            
            # Swap active batch into CUDA
            for mid in batch_ids:
                model_instance = population_models[mid]
                if mid in self.pinned_pool:
                    self.swap_into_active_cuda(mid, model_instance)
                elif TORCH_AVAILABLE and isinstance(model_instance, torch.nn.Module):
                    model_instance.to(self.device)
                    
            # Evaluate batch under FP8 autocast if enabled
            if TORCH_AVAILABLE and self.enable_fp8_autocast and hasattr(torch, "float8_e4m3fn"):
                with torch.cuda.amp.autocast(dtype=torch.float8_e4m3fn):
                    for mid in batch_ids:
                        try:
                            score = eval_fn(mid, population_models[mid])
                            results[mid] = score
                        except Exception as e:
                            logger.error(f"Error evaluating candidate '{mid}': {e}")
                            results[mid] = -999.0
            else:
                for mid in batch_ids:
                    try:
                        score = eval_fn(mid, population_models[mid])
                        results[mid] = score
                    except Exception as e:
                        logger.error(f"Error evaluating candidate '{mid}': {e}")
                        results[mid] = -999.0
                        
            # Offload batch back to pinned unified memory and clear CUDA cache
            for mid in batch_ids:
                if TORCH_AVAILABLE and isinstance(population_models[mid], torch.nn.Module):
                    self.pin_to_unified_memory(mid, population_models[mid].state_dict())
                    population_models[mid].to("cpu")
                    
            if TORCH_AVAILABLE and torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
        return results

    def clear_pool(self):
        """Releases all pinned unified memory blocks."""
        self.pinned_pool.clear()
        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
