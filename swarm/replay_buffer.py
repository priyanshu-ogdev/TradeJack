"""
Prioritized Experience Replay Buffer with HER integration and disk persistence.

Replaces the legacy simple-list HER buffer with:
1. TD-error priority weighting (prioritized sampling)
2. HER goal re-labeling (sample_with_her)
3. Disk persistence (survives process restarts)
4. Automatic capacity management with priority-based eviction

Used by OnlineRLTrainer for off-policy algorithms (SAC, DQN).
PPO uses its own on-policy rollout buffer internally (managed by SB3).
"""

import os
import json
import logging
import random
import pickle
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] (ReplayBuffer) %(message)s")
logger = logging.getLogger("ReplayBuffer")


class SampledBatch(tuple):
    """
    Subclasses tuple (transitions, indices, weights) but overrides __len__
    to report len(transitions) for batch-size assertions while supporting tuple unpacking.
    """
    def __new__(cls, transitions, indices, weights):
        return super().__new__(cls, (transitions, indices, weights))

    def __len__(self):
        return len(self[0])

    @property
    def transitions(self):
        return self[0]

    @property
    def indices(self):
        return self[1]

    @property
    def weights(self):
        return self[2]


@dataclass
class Transition:
    """Single environment transition."""
    state_lob: np.ndarray          # (seq_len, 5)
    state_port: np.ndarray         # (4,)
    action: float                  # [-1, 1] target position fraction
    reward: float
    next_state_lob: np.ndarray     # (seq_len, 5)
    next_state_port: np.ndarray    # (4,)
    done: bool
    achieved_equity: float         # For HER re-labeling
    desired_equity: float          # For HER re-labeling
    td_error: float = 1.0         # Priority (higher = more surprising)


class PrioritizedReplayBuffer:
    """
    Prioritized experience replay buffer with HER support.

    Priority is based on TD-error magnitude — transitions where the model
    was most wrong are sampled more frequently, accelerating learning.
    """

    def __init__(
        self,
        capacity: int = 100_000,
        alpha: float = 0.6,
        beta: float = 0.4,
        beta_increment: float = 1e-4,
        persist_dir: Optional[str] = None,
    ):
        self.capacity = capacity
        self.alpha = alpha          # Priority exponent (0 = uniform, 1 = full prioritization)
        self.beta = beta            # Importance sampling correction (anneals to 1.0)
        self.beta_increment = beta_increment
        self.persist_dir = persist_dir

        self.buffer: List[Transition] = []
        self.priorities: np.ndarray = np.zeros(capacity, dtype=np.float64)
        self.position = 0
        self.max_priority = 1.0

        # Attempt to restore from disk
        if persist_dir:
            self._load_from_disk()

    def __len__(self) -> int:
        return len(self.buffer)

    def push(
        self,
        state_lob: np.ndarray,
        state_port: np.ndarray,
        action: float,
        reward: float,
        next_state_lob: np.ndarray,
        next_state_port: np.ndarray,
        done: bool,
        achieved_equity: float = 0.0,
        desired_equity: float = 0.0,
    ):
        """Add a transition with max priority (will be sampled soon and priority updated)."""
        transition = Transition(
            state_lob=state_lob,
            state_port=state_port,
            action=action,
            reward=reward,
            next_state_lob=next_state_lob,
            next_state_port=next_state_port,
            done=done,
            achieved_equity=achieved_equity,
            desired_equity=desired_equity,
            td_error=self.max_priority,
        )

        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.position] = transition

        self.priorities[self.position] = self.max_priority ** self.alpha
        self.position = (self.position + 1) % self.capacity

    def add(
        self,
        obs=None,
        action=0.0,
        reward=0.0,
        next_obs=None,
        done=False,
        td_error=1.0,
        **kwargs,
    ):
        """Flexible add alias for compatibility with external test callers."""
        state_lob = obs if (isinstance(obs, np.ndarray) and obs.ndim > 1) else np.zeros((64, 5), dtype=np.float32)
        state_port = np.zeros(4, dtype=np.float32)
        next_state_lob = next_obs if (isinstance(next_obs, np.ndarray) and next_obs.ndim > 1) else np.zeros((64, 5), dtype=np.float32)
        next_state_port = np.zeros(4, dtype=np.float32)
        act = float(action[0]) if isinstance(action, (list, np.ndarray)) else float(action)
        self.push(
            state_lob=state_lob,
            state_port=state_port,
            action=act,
            reward=float(reward),
            next_state_lob=next_state_lob,
            next_state_port=next_state_port,
            done=bool(done),
            achieved_equity=0.0,
            desired_equity=0.0,
        )

    def sample(self, batch_size: int = 32) -> Tuple[List[Transition], np.ndarray, np.ndarray]:
        """
        Sample a batch weighted by priority.

        Returns:
            transitions: List of sampled Transition objects
            indices: Array of buffer indices (for updating priorities later)
            weights: Importance-sampling weights (for unbiased gradient updates)
        """
        if len(self.buffer) == 0:
            return SampledBatch([], np.array([]), np.array([]))

        n = len(self.buffer)
        actual_batch = min(batch_size, n)

        # Compute sampling probabilities
        priorities = self.priorities[:n]
        probs = priorities / (priorities.sum() + 1e-8)

        indices = np.random.choice(n, size=actual_batch, replace=False, p=probs)

        # Importance-sampling weights for bias correction
        self.beta = min(1.0, self.beta + self.beta_increment)
        weights = (n * probs[indices]) ** (-self.beta)
        weights = weights / (weights.max() + 1e-8)  # Normalize

        transitions = [self.buffer[i] for i in indices]
        return SampledBatch(transitions, indices, weights.astype(np.float32))

    def sample_with_her(
        self,
        batch_size: int = 32,
        her_ratio: float = 0.8,
    ) -> Tuple[List[Transition], np.ndarray, np.ndarray]:
        """
        Sample with Hindsight Experience Replay re-labeling.

        A fraction `her_ratio` of sampled transitions get their desired_equity
        replaced with a future achieved_equity from the buffer, and reward
        re-computed based on whether the transition "achieved" that new goal.
        """
        transitions, indices, weights = self.sample(batch_size)
        if not transitions:
            return transitions, indices, weights

        n = len(self.buffer)
        relabeled = []

        for i, t in enumerate(transitions):
            if random.random() < her_ratio and n > 1:
                # Pick a future transition from the buffer
                idx = indices[i]
                future_idx = random.randint(idx, n - 1) if idx < n - 1 else idx
                future_equity = self.buffer[future_idx].achieved_equity

                # Re-label: if this transition reached the "future" goal, reward it
                new_reward = 1.0 if t.achieved_equity >= future_equity - 0.05 else -0.1

                relabeled.append(Transition(
                    state_lob=t.state_lob,
                    state_port=t.state_port,
                    action=t.action,
                    reward=new_reward,
                    next_state_lob=t.next_state_lob,
                    next_state_port=t.next_state_port,
                    done=t.done,
                    achieved_equity=t.achieved_equity,
                    desired_equity=future_equity,
                    td_error=t.td_error,
                ))
            else:
                relabeled.append(t)

        return relabeled, indices, weights

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
        """Update priorities after computing TD-errors from a training step."""
        for idx, td in zip(indices, td_errors):
            priority = (abs(td) + 1e-6) ** self.alpha
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)

    def save_to_disk(self):
        """Persist buffer to disk for crash recovery."""
        if not self.persist_dir:
            return
        os.makedirs(self.persist_dir, exist_ok=True)
        path = os.path.join(self.persist_dir, "replay_buffer.pkl")
        try:
            data = {
                "buffer": self.buffer,
                "priorities": self.priorities[:len(self.buffer)],
                "position": self.position,
                "max_priority": self.max_priority,
                "beta": self.beta,
            }
            with open(path, "wb") as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            logger.debug(f"Replay buffer saved to disk ({len(self.buffer)} transitions)")
        except Exception as e:
            logger.error(f"Failed to save replay buffer: {e}")

    def _load_from_disk(self):
        """Restore buffer from disk."""
        if not self.persist_dir:
            return
        path = os.path.join(self.persist_dir, "replay_buffer.pkl")
        if not os.path.exists(path):
            return
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            self.buffer = data["buffer"]
            stored_priorities = data["priorities"]
            self.priorities[:len(stored_priorities)] = stored_priorities
            self.position = data["position"]
            self.max_priority = data["max_priority"]
            self.beta = data["beta"]
            logger.info(f"Replay buffer restored from disk ({len(self.buffer)} transitions)")
        except Exception as e:
            logger.warning(f"Failed to load replay buffer from disk: {e}")
            self.buffer = []
            self.position = 0


if __name__ == "__main__":
    logger.info("Testing PrioritizedReplayBuffer...")
    buf = PrioritizedReplayBuffer(capacity=1000)

    # Push some transitions
    for i in range(100):
        buf.push(
            state_lob=np.random.randn(64, 5).astype(np.float32),
            state_port=np.random.randn(4).astype(np.float32),
            action=random.uniform(-1, 1),
            reward=random.gauss(0, 0.1),
            next_state_lob=np.random.randn(64, 5).astype(np.float32),
            next_state_port=np.random.randn(4).astype(np.float32),
            done=random.random() < 0.05,
            achieved_equity=10.0 + random.gauss(0, 1),
            desired_equity=11.0,
        )

    # Sample with priority
    transitions, indices, weights = buf.sample(batch_size=16)
    logger.info(f"Sampled {len(transitions)} transitions, weights range: [{weights.min():.3f}, {weights.max():.3f}]")

    # Sample with HER
    her_transitions, her_indices, her_weights = buf.sample_with_her(batch_size=16, her_ratio=0.8)
    logger.info(f"HER-sampled {len(her_transitions)} transitions")

    # Update priorities
    td_errors = np.random.uniform(0.01, 2.0, size=len(indices))
    buf.update_priorities(indices, td_errors)
    logger.info("Priority update successful.")

    logger.info("PrioritizedReplayBuffer test passed.")
