"""
SB3ReplayBufferAdapter: makes swarm/replay_buffer.py's PrioritizedReplayBuffer
usable as SAC/DQN's actual replay buffer, via SB3's `replay_buffer_class` /
`replay_buffer_kwargs` construction hook.

Why this didn't just work by passing PrioritizedReplayBuffer directly: SB3's
off-policy algorithms (SAC, DQN) call very specific methods with a very
specific contract —
    buffer.add(obs: dict[str, np.ndarray], next_obs: dict[str, np.ndarray],
               action: np.ndarray, reward: np.ndarray, done: np.ndarray,
               infos: list[dict]) -> None
    buffer.sample(batch_size: int, env=None) -> DictReplayBufferSamples
        (a NamedTuple of torch.Tensors: observations, actions,
         next_observations, dones, rewards, discounts)
    buffer.size() -> int
PrioritizedReplayBuffer's own `push()`/`sample()` predate that contract — they
work with `Transition` dataclasses and split lob/portfolio arguments, not a
single obs dict, and `sample()` returns a `SampledBatch` of `Transition`
objects, not a tensor-based NamedTuple. Passing it directly as
`replay_buffer_class` would build without error (Python doesn't check method
signatures at construction time) and then fail — or worse, silently
misbehave — the moment SB3 actually called `.add()`/`.sample()` during
training. This class is the missing translation layer, not a modification of
either side's existing contract.

Design choice: inherit from DictReplayBuffer for type-compatibility
(isinstance checks, type hints elsewhere in SB3) but deliberately skip its
__init__ (which pre-allocates buffer_size-sized numpy arrays for every obs/
action/reward field — at SB3's default buffer_size=1_000_000 that's a
meaningful chunk of memory allocated and then never used, since storage is
fully delegated to the wrapped PrioritizedReplayBuffer instead). We call
BaseBuffer.__init__ (the lightweight grandparent) instead.
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from stable_baselines3.common.buffers import DictReplayBuffer, BaseBuffer
from stable_baselines3.common.type_aliases import DictReplayBufferSamples

from swarm.replay_buffer import PrioritizedReplayBuffer

logger = logging.getLogger("SB3ReplayBufferAdapter")


class SB3ReplayBufferAdapter(DictReplayBuffer):
    def __init__(
        self,
        buffer_size: int,
        observation_space,
        action_space,
        device="auto",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = True,
        her_ratio: float = 0.8,
        alpha: float = 0.6,
        beta: float = 0.4,
        persist_dir: Optional[str] = None,
    ):
        # Deliberately BaseBuffer.__init__, not DictReplayBuffer.__init__ —
        # see module docstring for why. This sets self.buffer_size,
        # self.observation_space, self.action_space, self.device, self.n_envs
        # without pre-allocating any storage arrays, since PrioritizedReplayBuffer
        # is where storage actually lives.
        BaseBuffer.__init__(self, buffer_size, observation_space, action_space, device, n_envs=n_envs)

        if n_envs != 1:
            logger.warning(
                f"SB3ReplayBufferAdapter got n_envs={n_envs}, but PrioritizedReplayBuffer "
                "was designed around a single env. Multi-env vectorized training will store "
                "transitions from all envs into one buffer without per-env bookkeeping — "
                "functionally fine, just not what per-env replay analysis would expect."
            )

        self.her_ratio = her_ratio
        self._buf = PrioritizedReplayBuffer(
            capacity=buffer_size, alpha=alpha, beta=beta, persist_dir=persist_dir
        )

    def add(
        self,
        obs: Dict[str, np.ndarray],
        next_obs: Dict[str, np.ndarray],
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: List[Dict[str, Any]],
    ) -> None:
        """SB3's actual call contract for off-policy add() — called once per
        env step during collect_rollouts(). Unwraps the (n_envs, ...) batch
        dimension SB3 always includes (even for n_envs=1) down to the plain
        per-transition arguments PrioritizedReplayBuffer.push() expects."""
        n = action.shape[0] if hasattr(action, "shape") and action.ndim > 0 else 1
        for i in range(n):
            achieved_equity = float(obs["portfolio_state"][i][2]) if "portfolio_state" in obs else 0.0
            self._buf.push(
                state_lob=np.asarray(obs["lob_sequence"][i]),
                state_port=np.asarray(obs["portfolio_state"][i]),
                action=float(np.asarray(action[i]).reshape(-1)[0]),
                reward=float(np.asarray(reward[i]).reshape(-1)[0]),
                next_state_lob=np.asarray(next_obs["lob_sequence"][i]),
                next_state_port=np.asarray(next_obs["portfolio_state"][i]),
                done=bool(np.asarray(done[i]).reshape(-1)[0]),
                achieved_equity=achieved_equity,
                desired_equity=achieved_equity,  # no explicit goal signal upstream yet; HER relabels this in sample()
            )

    def sample(self, batch_size: int, env=None) -> DictReplayBufferSamples:
        """
        SB3's actual call contract — must return a DictReplayBufferSamples of
        torch.Tensors, not PrioritizedReplayBuffer's native SampledBatch of
        Transition dataclasses. This is the real translation step: sample with
        HER re-labeling using the buffer's own logic, then repack the result
        into the exact tensor shapes SAC/DQN's train() indexes into directly
        (samples.rewards, samples.dones, etc. — get a field wrong here and
        training silently learns from the wrong signal instead of crashing,
        which is why every field below is built explicitly rather than
        inferred).
        """
        if len(self._buf) == 0:
            zero_obs = {
                "lob_sequence": torch.zeros((0, *self.observation_space["lob_sequence"].shape), device=self.device),
                "portfolio_state": torch.zeros((0, *self.observation_space["portfolio_state"].shape), device=self.device),
            }
            return DictReplayBufferSamples(
                observations=zero_obs, actions=torch.zeros((0, 1), device=self.device),
                next_observations=zero_obs, dones=torch.zeros((0, 1), device=self.device),
                rewards=torch.zeros((0, 1), device=self.device), discounts=torch.zeros((0, 1), device=self.device),
            )

        transitions, indices, weights = self._buf.sample_with_her(batch_size, her_ratio=self.her_ratio)
        self._last_sampled_indices = indices  # update_priorities() needs these after a TD-error pass

        obs = {
            "lob_sequence": torch.as_tensor(np.stack([t.state_lob for t in transitions]), dtype=torch.float32, device=self.device),
            "portfolio_state": torch.as_tensor(np.stack([t.state_port for t in transitions]), dtype=torch.float32, device=self.device),
        }
        next_obs = {
            "lob_sequence": torch.as_tensor(np.stack([t.next_state_lob for t in transitions]), dtype=torch.float32, device=self.device),
            "portfolio_state": torch.as_tensor(np.stack([t.next_state_port for t in transitions]), dtype=torch.float32, device=self.device),
        }
        actions = torch.as_tensor(np.array([[t.action] for t in transitions]), dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(np.array([[float(t.done)] for t in transitions]), dtype=torch.float32, device=self.device)
        rewards = torch.as_tensor(np.array([[t.reward] for t in transitions]), dtype=torch.float32, device=self.device)
        discounts = torch.ones_like(rewards)  # no per-transition discount signal upstream; SAC/DQN apply gamma separately

        return DictReplayBufferSamples(
            observations=obs, actions=actions, next_observations=next_obs,
            dones=dones, rewards=rewards, discounts=discounts,
        )

    def update_priorities_from_last_sample(self, td_errors: np.ndarray):
        """Not part of SB3's contract — SB3 never calls this itself, so TD-error
        priority updates need an explicit hook. Call this after a train() step
        if you have real TD-errors (e.g. from a custom callback reading the
        critic's loss per-sample); without it, priorities decay toward uniform
        sampling over time as push()'s max-priority initialization gets diluted,
        which degrades to "prioritized in name only" — silently, not with an error."""
        if hasattr(self, "_last_sampled_indices"):
            self._buf.update_priorities(self._last_sampled_indices, td_errors)
        else:
            logger.warning("update_priorities_from_last_sample() called before any sample() — nothing to update.")

    def size(self) -> int:
        return len(self._buf)

    def __len__(self) -> int:
        return len(self._buf)


if __name__ == "__main__":
    print("Smoke-testing SB3ReplayBufferAdapter with a real DQN.learn() call...")
    import gymnasium as gym
    from gymnasium import spaces
    import sys, os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

    class FakeEnv(gym.Env):
        def __init__(self):
            super().__init__()
            self.action_space = spaces.Discrete(3)
            self.observation_space = spaces.Dict({
                "lob_sequence": spaces.Box(low=-np.inf, high=np.inf, shape=(64, 5), dtype=np.float32),
                "portfolio_state": spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32),
            })
            self.t = 0

        def reset(self, seed=None, options=None):
            self.t = 0
            return {"lob_sequence": np.random.randn(64, 5).astype(np.float32),
                    "portfolio_state": np.random.randn(4).astype(np.float32)}, {}

        def step(self, action):
            self.t += 1
            obs = {"lob_sequence": np.random.randn(64, 5).astype(np.float32),
                   "portfolio_state": np.random.randn(4).astype(np.float32)}
            return obs, float(np.random.randn() * 0.01), self.t >= 50, False, {}

    from stable_baselines3 import DQN
    from swarm.shared_encoder import LOBFeatureExtractor

    env = FakeEnv()
    model = DQN(
        "MultiInputPolicy", env, device="cpu", buffer_size=2000, learning_starts=32,
        replay_buffer_class=SB3ReplayBufferAdapter,
        replay_buffer_kwargs=dict(her_ratio=0.8),
        policy_kwargs=dict(features_extractor_class=LOBFeatureExtractor, features_extractor_kwargs=dict(features_dim=64)),
        verbose=0,
    )
    print(f"Buffer type: {type(model.replay_buffer).__name__}")
    assert isinstance(model.replay_buffer, SB3ReplayBufferAdapter)

    params_before = [p.clone().detach() for p in model.policy.parameters()]
    model.learn(total_timesteps=256)
    params_after = list(model.policy.parameters())

    changed = any(not torch.allclose(b, a) for b, a in zip(params_before, params_after))
    print(f"Buffer size after training: {model.replay_buffer.size()}")
    print(f"Policy weights actually changed: {changed}")
    assert model.replay_buffer.size() > 0, "adapter's add() should have stored real transitions"
    assert changed, "DQN.learn() should have produced real gradient updates sampling from our buffer"

    batch = model.replay_buffer.sample(16)
    print(f"Sampled batch shapes: obs={batch.observations['lob_sequence'].shape}, actions={batch.actions.shape}, rewards={batch.rewards.shape}")
    assert batch.observations["lob_sequence"].shape[0] == 16

    print("SB3ReplayBufferAdapter smoke test passed: real DQN training used our prioritized+HER buffer end to end.")
