"""
Reinforcement Learning & Evolutionary Mechanics (`rl_mechanics.py`).
Implements:
1. Hindsight Experience Replay (HER): Re-labels failed equity trajectories using achieved terminal equity as goals.
2. Population Based Training (PBT): Exploits top 20% surviving lineage weights and mutates bottom 20% parameters.
3. Adversarial GAN Spoofer: Injects synthetic order book spoofing patterns during training for resilience against manipulation.
"""

import os
import sys
import time
import math
import random
import logging
import numpy as np
from typing import Dict, Any, List, Optional, Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] (RLMechanics) %(message)s")
logger = logging.getLogger("RLMechanics")

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class HindsightExperienceReplay:
    """
    HER Buffer allowing micro-scalping agents to learn from failed or stagnant trajectories.
    Stores transitions and re-labels desired equity goals with achieved intermediate states.
    """

    def __init__(self, capacity: int = 50000, goal_strategy: str = "future"):
        self.capacity = capacity
        self.goal_strategy = goal_strategy
        self.buffer: List[Dict[str, Any]] = []
        self.position = 0

    def push(
        self,
        state: np.ndarray,
        action: Any,
        reward: float,
        next_state: np.ndarray,
        achieved_equity: float,
        desired_equity: float,
        done: bool
    ):
        transition = {
            "state": state,
            "action": action,
            "reward": reward,
            "next_state": next_state,
            "achieved_equity": achieved_equity,
            "desired_equity": desired_equity,
            "done": done
        }
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.position] = transition
        self.position = (self.position + 1) % self.capacity

    def sample_with_her(self, batch_size: int = 32, her_ratio: float = 0.8) -> List[Dict[str, Any]]:
        """
        Samples a batch where `her_ratio` fraction of transitions are re-labeled with achieved future goals.
        """
        if not self.buffer:
            return []
            
        actual_batch_size = min(batch_size, len(self.buffer))
        sampled_indices = np.random.choice(len(self.buffer), actual_batch_size, replace=False)
        batch = []
        
        for idx in sampled_indices:
            transition = dict(self.buffer[idx])
            # Check if we should re-label goal
            if random.random() < her_ratio and len(self.buffer) > 1:
                # Sample a future state from buffer (or any random state if short)
                future_idx = random.randint(idx, len(self.buffer) - 1)
                future_achieved = self.buffer[future_idx]["achieved_equity"]
                
                # Re-compute reward under new goal: if achieved >= future_achieved, positive reward
                transition["desired_equity"] = future_achieved
                if transition["achieved_equity"] >= future_achieved - 0.05:
                    transition["reward"] = 1.0
                else:
                    transition["reward"] = -0.1
            batch.append(transition)
        return batch

    def __len__(self) -> int:
        return len(self.buffer)


class PopulationBasedTrainingEngine:
    """
    Evaluates fitness across the sovereign swarm population, exploiting top performers and mutating bottom performers.
    """

    def __init__(self, swarm_size: int = 50, exploit_fraction: float = 0.2):
        self.swarm_size = swarm_size
        self.exploit_fraction = exploit_fraction

    def execute_pbt_step(self, population_status: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Takes `population_status` (list of dicts with `child_id`, `equity`, `sharpe_ratio`, `weights_path`, `learning_rate`),
        sorts by fitness (Sharpe * Equity), and mutates the bottom 20% by copying top 20% weights + noise.
        """
        if len(population_status) < 2:
            return population_status
            
        # Sort by fitness descending
        sorted_pop = sorted(population_status, key=lambda x: x.get("sharpe_ratio", 0.0) * x.get("equity", 10.0), reverse=True)
        
        cutoff_count = max(1, int(len(sorted_pop) * self.exploit_fraction))
        top_performers = sorted_pop[:cutoff_count]
        bottom_performers = sorted_pop[-cutoff_count:]
        
        logger.info(f"PBT Engine executing cycle across {len(sorted_pop)} containers. Top cutoff: {cutoff_count}.")
        
        for bottom_child in bottom_performers:
            parent = random.choice(top_performers)
            logger.info(f"[PBT EXPLOIT/EXPLORE] Child {bottom_child['child_id']} inheriting weights from Top Parent {parent['child_id']}.")
            
            # Mutate hyperparameters
            old_lr = bottom_child.get("learning_rate", 0.001)
            mutation_factor = random.uniform(0.8, 1.2)
            bottom_child["learning_rate"] = old_lr * mutation_factor
            bottom_child["parent_lineage"] = parent["child_id"]
            
            # Copy parent weights file if exists
            parent_weights = parent.get("weights_path")
            child_weights = bottom_child.get("weights_path")
            if parent_weights and child_weights and os.path.exists(parent_weights) and parent_weights != child_weights:
                try:
                    import shutil
                    shutil.copy2(parent_weights, child_weights)
                except Exception as e:
                    logger.debug(f"Failed copying parent weights: {e}")
                    
        return sorted_pop


class AdversarialGANSpoofer:
    """
    Lightweight order-flow spoofing generator injecting artificial imbalance spikes and wash trading patterns
    during local training so that Child agents learn robust defenses against market manipulation.
    """

    def __init__(self, spoof_intensity: float = 0.5):
        self.spoof_intensity = spoof_intensity

    def inject_spoof_noise(self, lob_sequence: np.ndarray) -> np.ndarray:
        """
        Takes a rolling LOB sequence `[seq_len, 8]` (`bp0, bs0, ap0, as0, bp1, bs1, ap1, as1`)
        and injects transient fake volume walls on one side followed by sudden cancellation.
        """
        if lob_sequence is None or len(lob_sequence) == 0:
            return lob_sequence
            
        spoofed = np.copy(lob_sequence)
        seq_len = spoofed.shape[0]
        
        # Randomly choose to spoof bid wall or ask wall for 5 consecutive steps
        if random.random() < self.spoof_intensity and seq_len > 10:
            start_t = random.randint(0, seq_len - 6)
            side = random.choice(["bid", "ask"])
            
            if side == "bid":
                # Inflate bid size 0 (feature index 1) by 10x for 5 steps, then collapse
                spoofed[start_t : start_t + 5, 1] *= 10.0
            else:
                # Inflate ask size 0 (feature index 3) by 10x for 5 steps, then collapse
                spoofed[start_t : start_t + 5, 3] *= 10.0
                
        return spoofed


if __name__ == "__main__":
    logger.info("Testing RLMechanics components standalone...")
    her = HindsightExperienceReplay(capacity=100)
    her.push(np.zeros(8), [0.5], -0.1, np.ones(8), achieved_equity=9.5, desired_equity=11.0, done=False)
    her.push(np.zeros(8), [0.2], 1.0, np.ones(8), achieved_equity=11.2, desired_equity=11.0, done=True)
    print("HER sampled batch len:", len(her.sample_with_her(batch_size=2)))
    
    pbt = PopulationBasedTrainingEngine(swarm_size=5, exploit_fraction=0.4)
    status = [
        {"child_id": 1, "equity": 15.0, "sharpe_ratio": 2.5, "learning_rate": 0.001},
        {"child_id": 2, "equity": 8.0,  "sharpe_ratio": -0.5, "learning_rate": 0.001}
    ]
    print("PBT result:", json.dumps(pbt.execute_pbt_step(status), indent=2))
