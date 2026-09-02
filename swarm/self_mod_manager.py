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
import ast
import subprocess
import shutil
import importlib.util
import gc
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
from swarm.model_registry import REGISTRY, ModelCard

# v3: Legacy model class imports removed — the 24 educational toy ports from
# huseinzol05/Stock-Prediction-Models have been replaced with 5 SB3-backed
# models. SelfModEngine now manages SB3 algorithm selection, not raw nn.Module
# architecture swapping.

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


class SelfModEngine:
    """
    Manages self-modification and universal model architecture transitions inside Docker.
    Backed by `TradeJackModelRegistry` containing all 20+ Stock-Prediction-Models architectures.
    """

    def __init__(self, child_id: int = 0, state_dir: str = "d:/TradeJack/state"):
        self.child_id = child_id
        self.state_dir = os.path.abspath(state_dir)
        self.active_model: Any = None  # v3: model created via OnlineRLTrainer, not directly from registry
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

    def select_optimal_model_for_survival(self, current_tier: int, current_sharpe: float, current_equity: float) -> str:
        """
        Dynamically selects the optimal SB3-backed model name given survival mode and memory tier.

        v3 model mapping:
          Tier 1 (20GB): PPO-Transformer (high-capacity, long-horizon)
          Tier 2 (4GB):  PPO-DilatedCNN or SAC-DilatedCNN
          Tier 3 (CPU):  DuelingDQN (lightweight, fast iteration)
        """
        if current_tier == 3 or current_equity < 5.00:
            return "DuelingDQN"
        elif current_tier == 1 and current_sharpe > 2.0 and current_equity >= 20.0:
            return "PPO-Transformer"
        elif current_tier == 1 and current_sharpe > 1.5:
            return "SAC-DilatedCNN"
        else:
            return "PPO-DilatedCNN" if current_sharpe >= 0.0 else "SAC-DilatedCNN"

    def swap_active_architecture(
        self,
        target_model_name: str,
        current_tier: int,
        calibration_loader: Optional[Any] = None,
        input_dim: int = 5
    ) -> Any:
        """
        Enforces Warden memory boundaries across all 20+ architectures in TradeJackModelRegistry while computing EWC bounds.
        """
        self.active_tier = current_tier
        logger.info(f"SelfMod Engine triggered model swap to '{target_model_name}' under Tier {current_tier} rules.")
        
        card = REGISTRY.get_model_card(target_model_name)
        if not card:
            logger.warning(f"Model '{target_model_name}' not found in registry. Using fallback.")
            card = REGISTRY.get_model_card("Deep-Q-learning")
            target_model_name = "Deep-Q-learning"
            
        # Enforce Warden limits: tier_requirement is the minimum memory capability needed (1=20GB, 2=4GB, 3=1GB).
        # If current_tier > card.tier_requirement, the container has less capability than needed!
        if current_tier > card.tier_requirement:
            logger.warning(f"Tier {current_tier} forbids model '{target_model_name}' (requires Tier {card.tier_requirement}). Downgrading to safe tier.")
            allowed_cards = REGISTRY.list_models_for_tier(max_tier=current_tier)
            if allowed_cards:
                # Pick the first/best allowed model for this tier
                card = allowed_cards[0]
                target_model_name = card.model_name
                if current_tier == 3 and target_model_name != "Deep-Q-learning":
                    card = REGISTRY.get_model_card("Deep-Q-learning")
                    target_model_name = "Deep-Q-learning"
                elif current_tier == 2 and target_model_name == "Attention-is-all-you-Need":
                    card = REGISTRY.get_model_card("Dilated-CNN-Seq2seq")
                    target_model_name = "Dilated-CNN-Seq2seq"
            else:
                card = REGISTRY.get_model_card("Deep-Q-learning")
                target_model_name = "Deep-Q-learning"

        # Compute EWC on outgoing active model before destroying or offloading
        if self.active_model is not None and calibration_loader is not None:
            logger.info("Computing EWC Fisher matrix on outgoing model...")
            self.ewc_instance = ElasticWeightConsolidation(self.active_model, calibration_loader)
            
        # SOTA Fix Bug 2: Aggressive CUDA Purge to prevent Memory Creep
        if self.active_model is not None:
            if hasattr(self.active_model, 'cpu'):
                self.active_model.cpu()
            del self.active_model
            gc.collect()
            if TORCH_AVAILABLE and torch.cuda.is_available():
                torch.cuda.empty_cache()
            if "active_model" in sys.modules:
                del sys.modules["active_model"]
                
        new_model = REGISTRY.build_model(target_model_name, input_dim=input_dim)
        self.active_model = new_model
        return new_model

    def invoke_vllm_reasoning_bridge(self, current_tier: int, stagnation_duration: float, current_sharpe: float, input_dim: int = 5):
        """
        Mock vLLM reasoning bridge to simulate calling the Repo 2 Automaton Node.js sidecar.
        Informs the Automaton of the stagnation, and requests a rewritten PyTorch architecture
        bound by the current Warden tier allocations.
        Implements SOTA AST Sandboxing, Subprocess Timeout, and ImportLib hot-swapping.
        """
        logger.warning(f"=== Automaton vLLM Reasoning Bridge Invoked ===")
        logger.warning(f"  VRAM Tier: {current_tier} (Constraint)")
        logger.warning(f"  Stagnation: {stagnation_duration/3600:.2f} hours")
        logger.warning(f"  Current Sharpe: {current_sharpe:.2f}")
        logger.warning(f"  Prompting local LLM endpoint to rewrite {self.active_model.model_name}...")
        
        time.sleep(1.0)
        logger.info(f"Automaton responded. Injecting Blackwell FP8 optimizations...")
        
        mock_new_code = f'''
import torch
import torch.nn as nn
import numpy as np

class AutonomousModel(nn.Module):
    def __init__(self, input_dim={input_dim}, hidden_dim=64):
        super().__init__()
        self.model_name = "VLLM-Autonomous-Model"
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        if x.dim() == 3: x = torch.mean(x, dim=1)
        elif x.dim() == 1: x = x.unsqueeze(0)
        return self.fc2(torch.relu(self.fc1(x)))
'''
        mock_new_code = self.inject_blackwell_fp8_optimizations(mock_new_code)
        
        # 1. Save to Staging (Fix PyTorch File-Lock Contention)
        child_dir = os.path.join(self.state_dir, f"child_{self.child_id}")
        os.makedirs(child_dir, exist_ok=True)
        staging_path = os.path.join(child_dir, "staging_model.py")
        active_path = os.path.join(child_dir, "active_model.py")
        
        with open(staging_path, "w", encoding="utf-8") as f:
            f.write(mock_new_code)
            
        # 2. AST Parse (Fix Hallucinated Syntax Crash)
        try:
            ast.parse(mock_new_code)
        except SyntaxError as e:
            logger.error(f"Hallucinated Syntax Error in LLM Code: {e}")
            return
            
        # 3. Sandboxed Dry-Run (Fix Runaway Shell Infinite Loop)
        dry_run_script = f'''
import torch
import sys
import os
sys.path.insert(0, r'{child_dir}')
import staging_model
model = staging_model.AutonomousModel(input_dim={input_dim})
x = torch.randn(1, 64, {input_dim})
out = model.forward(x)
if out.shape[0] != 1:
    sys.exit(1)
'''
        dry_path = os.path.join(child_dir, "dry_run.py")
        with open(dry_path, "w", encoding="utf-8") as f:
            f.write(dry_run_script)
            
        try:
            logger.info("Initiating Sandboxed Dry-Run...")
            subprocess.run([sys.executable, dry_path], timeout=15.0, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except subprocess.TimeoutExpired:
            logger.error("Runaway Shell Infinite Loop detected! Terminating subprocess.")
            return
        except subprocess.CalledProcessError:
            logger.error("Dry-Run crashed (Shape mismatch or Runtime Error)!")
            return
            
        # 4. Atomic File-Swap (No file locks during live forward passes)
        shutil.move(staging_path, active_path)
        
        # 5. Dynamic Module Loading (Ghost Brain Fix)
        # SOTA Fix Bug 2: Aggressive CUDA Purge to prevent Memory Creep
        if self.active_model is not None:
            if hasattr(self.active_model, 'cpu'):
                self.active_model.cpu()
            del self.active_model
            gc.collect()
            if TORCH_AVAILABLE and torch.cuda.is_available():
                torch.cuda.empty_cache()
            if "active_model" in sys.modules:
                del sys.modules["active_model"]

        spec = importlib.util.spec_from_file_location("active_model", active_path)
        if spec and spec.loader:
            active_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(active_module)
            self.active_model = active_module.AutonomousModel(input_dim=input_dim)
            logger.info(f"Ghost Brain bypassed. Active model swapped to {self.active_model.model_name} dynamically.")


if __name__ == "__main__":
    logger.info("Testing SelfModEngine standalone execution...")
    engine = SelfModEngine(child_id=1)
    print("Initial active model:", engine.active_model.model_name)
    
    # Test tier 2 restriction against transformer
    engine.swap_active_architecture("Attention-is-all-you-Need", current_tier=2)
    print("Active model after Tier 2 swap request:", engine.active_model.model_name)
    
    # Test tier 1 allowance
    engine.swap_active_architecture("Attention-is-all-you-Need", current_tier=1)
    print("Active model after Tier 1 swap request:", engine.active_model.model_name)
    
    # Test survival model selection
    opt = engine.select_optimal_model_for_survival(current_tier=3, current_sharpe=-0.5, current_equity=2.50)
    print("Optimal model for critical survival mode:", opt)
    
    # Test VLLM Reasoning Bridge
    engine.invoke_vllm_reasoning_bridge(current_tier=1, stagnation_duration=15000.0, current_sharpe=-1.0, input_dim=5)
    print("Active model after vLLM rewrite:", engine.active_model.model_name)
