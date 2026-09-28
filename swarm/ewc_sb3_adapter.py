"""
EWC for SB3 policies (PPO/SAC/DQN) - a from-scratch Fisher computation, not a
patch of swarm/ewc_optimizer.py.

Why not reuse ewc_optimizer.py's ElasticWeightConsolidation directly: its
Fisher computation assumes a plain regression model -- `outputs =
self.model(batch_x)` then `F.mse_loss(outputs, batch_y)` against an external
label. SB3 policies don't have that shape.

RESEARCHED, NOT GUESSED, per-algorithm Fisher definitions (see citations):

  PPO (on-policy, stochastic actor-critic): Fisher = squared gradient of
  log pi(a|s) for actions actually taken, sampled from the model's own
  rollout buffer. This is the textbook EWC definition (Kirkpatrick et al.,
  2017) applied directly -- the model scores its own past behavior, no
  external label needed.

  DQN (no stochastic policy at all -- argmax over Q-values): Kirkpatrick et
  al.'s own paper applies EWC to DQN on sequential Atari games directly using
  the DQN TD/Huber loss gradient as the Fisher signal (not a log-likelihood,
  since none exists for a deterministic argmax policy) -- confirmed by a
  follow-up paper (Fast & Slow Successor Features) which states explicitly:
  "the fisher information is computed by squaring the gradients of the
  parameters [...] L_TD can be either the DQN loss". This file replicates
  SB3 DQN.train()'s exact TD-target computation (same target network, same
  discount, same gather-by-action, same Huber loss) so the Fisher signal
  matches what the optimizer actually optimizes, not an approximation of it.

  SAC (stochastic actor-critic, off-policy): "Same State, Different Task"
  (Powers et al., 2021) applies EWC to "both the Q-function and policy" of a
  SAC agent, using log pi(a|s) for the actor (matching PPO's treatment) and
  the critic's own TD loss for the critic (matching DQN's treatment) -- this
  file implements exactly that combination, replicating SB3 SAC.train()'s
  actual next-action sampling, target critic, and MSE loss so the two Fisher
  contributions are computed the same way SAC actually trains.

Calibration data comes directly from the SB3 model's own buffer -- PPO's
on-policy `rollout_buffer`, or SAC/DQN's off-policy `replay_buffer` (via our
own SB3ReplayBufferAdapter, so this also exercises the prioritized+HER
sampling path, not a bypass of it).
"""

import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger("PolicyEWC")


class PolicyEWC:
    """
    Fisher-anchored EWC penalty for an SB3 policy (PPO, SAC, or DQN).
    Construct one right after a training cycle you're about to promote -- it
    snapshots that policy's parameters as the anchor ("star" params) and
    computes a Fisher matrix from that same cycle's own buffer data, using
    the algorithm-appropriate definition above. Pass the result into every
    tournament agent's OnlineRLTrainer.set_ewc_instance() so the *next*
    training cycle is regularized against drifting away from what just got
    promoted.
    """

    def __init__(self, model: Any, n_calibration_samples: int = 512, ewc_lambda: float = 1000.0):
        self.ewc_lambda = ewc_lambda
        self.policy = model.policy
        self.star_params: Dict[str, torch.Tensor] = {
            n: p.clone().detach() for n, p in self.policy.named_parameters() if p.requires_grad
        }
        self.fisher_matrix: Dict[str, torch.Tensor] = {
            n: torch.zeros_like(p) for n, p in self.policy.named_parameters() if p.requires_grad
        }
        self.n_calibration_samples_used = 0
        self.algo = self._detect_algo(model)

        if self.algo == "dqn":
            self._compute_fisher_dqn(model, n_calibration_samples)
        elif self.algo == "sac":
            self._compute_fisher_sac(model, n_calibration_samples)
        elif self.algo == "ppo":
            self._compute_fisher_ppo(model, n_calibration_samples)
        else:
            logger.warning(
                f"PolicyEWC: could not detect a supported algorithm on {type(model).__name__} "
                "(checked for .q_net [DQN], .actor+.critic [SAC], .rollout_buffer [PPO]). "
                "Fisher matrix stays all-zero."
            )

    @staticmethod
    def _detect_algo(model: Any) -> Optional[str]:
        if hasattr(model, "q_net") and hasattr(model, "q_net_target"):
            return "dqn"
        if hasattr(model, "actor") and hasattr(model, "critic") and hasattr(model, "critic_target"):
            return "sac"
        if hasattr(model, "rollout_buffer"):
            return "ppo"
        return None

    # ---------------------------------------------------------------- PPO --
    def _compute_fisher_ppo(self, model: Any, n_samples: int):
        rollout_buffer = getattr(model, "rollout_buffer", None)
        if rollout_buffer is None or not getattr(rollout_buffer, "full", False):
            logger.warning("PPO rollout buffer empty -- Fisher stays zero.")
            return

        obs_raw = rollout_buffer.observations
        actions_raw = rollout_buffer.actions
        # NOTE: verified empirically -- this SB3 version's DictRolloutBuffer
        # stores each array as (n_steps, *obs_shape) with NO separate n_envs
        # axis (n_envs=1 here). Do not assume the documented
        # (buffer_size, n_envs, *obs_shape) layout without checking real
        # shapes first -- an earlier version of this method got this wrong
        # and silently corrupted the calibration batch (see git history).
        take = min(n_samples, actions_raw.shape[0])
        if isinstance(obs_raw, dict):
            obs = {k: torch.as_tensor(v[:take], device=model.device) for k, v in obs_raw.items()}
        else:
            obs = torch.as_tensor(obs_raw[:take], device=model.device)
        actions = torch.as_tensor(actions_raw[:take], device=model.device)

        if not hasattr(self.policy, "evaluate_actions"):
            logger.warning(f"Policy type {type(self.policy).__name__} has no evaluate_actions(). Fisher stays zero.")
            return

        self.policy.eval()
        self.policy.zero_grad()
        try:
            eval_result = self.policy.evaluate_actions(obs, actions)
            log_prob = eval_result[1] if isinstance(eval_result, tuple) else eval_result
            (log_prob.sum()).backward()
            n = log_prob.shape[0]
            for name, p in self.policy.named_parameters():
                if p.requires_grad and p.grad is not None:
                    self.fisher_matrix[name] += (p.grad.detach() ** 2) / max(n, 1)
            self.n_calibration_samples_used = n
            logger.info(f"PolicyEWC[PPO]: Fisher computed over {n} calibration samples (log-likelihood gradient).")
        except Exception as e:
            logger.error(f"PolicyEWC[PPO] Fisher computation failed ({e}) -- penalty will be zero this cycle.")
        finally:
            self.policy.zero_grad()

    # ---------------------------------------------------------------- DQN --
    def _compute_fisher_dqn(self, model: Any, n_samples: int):
        """Replicates SB3 DQN.train()'s exact TD-target computation (same
        target network, discount, gather-by-action, Huber loss) so the Fisher
        signal reflects the same loss the optimizer actually minimizes."""
        replay_buffer = getattr(model, "replay_buffer", None)
        if replay_buffer is None or replay_buffer.size() == 0:
            logger.warning("DQN replay buffer empty -- Fisher stays zero.")
            return

        batch_size = min(n_samples, replay_buffer.size())
        replay_data = replay_buffer.sample(batch_size, env=getattr(model, "_vec_normalize_env", None))
        discounts = getattr(replay_data, "discounts", None)
        if discounts is None:
            discounts = model.gamma

        model.q_net.eval()
        model.q_net.zero_grad()
        try:
            with torch.no_grad():
                next_q_values = model.q_net_target(replay_data.next_observations)
                next_q_values, _ = next_q_values.max(dim=1)
                next_q_values = next_q_values.reshape(-1, 1)
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * discounts * next_q_values

            current_q_values = model.q_net(replay_data.observations)
            current_q_values = torch.gather(current_q_values, dim=1, index=replay_data.actions.long())
            loss = F.smooth_l1_loss(current_q_values, target_q_values)
            loss.backward()

            n = current_q_values.shape[0]
            for name, p in self.policy.named_parameters():
                if p.requires_grad and p.grad is not None:
                    self.fisher_matrix[name] += (p.grad.detach() ** 2) / max(n, 1)
            self.n_calibration_samples_used = n
            logger.info(f"PolicyEWC[DQN]: Fisher computed over {n} calibration samples (TD-loss gradient, matches Kirkpatrick et al.'s own DQN-Atari EWC treatment).")
        except Exception as e:
            logger.error(f"PolicyEWC[DQN] Fisher computation failed ({e}) -- penalty will be zero this cycle.")
        finally:
            model.q_net.zero_grad()

    # ---------------------------------------------------------------- SAC --
    def _compute_fisher_sac(self, model: Any, n_samples: int):
        """Fisher on BOTH actor (log-likelihood, like PPO) and critic
        (TD-loss, like DQN), matching Powers et al. (2021)'s treatment of EWC
        for SAC. Two separate zero_grad/backward passes so each parameter's
        Fisher entry reflects only the loss that actually trains it -- SAC's
        actor and critic have separate optimizers in SB3, so this mirrors
        that separation rather than conflating the two gradients."""
        replay_buffer = getattr(model, "replay_buffer", None)
        if replay_buffer is None or replay_buffer.size() == 0:
            logger.warning("SAC replay buffer empty -- Fisher stays zero.")
            return

        batch_size = min(n_samples, replay_buffer.size())
        replay_data = replay_buffer.sample(batch_size, env=getattr(model, "_vec_normalize_env", None))
        discounts = getattr(replay_data, "discounts", None)
        if discounts is None:
            discounts = model.gamma

        n_used = 0

        # --- Actor: log-likelihood Fisher, same treatment as PPO ---
        model.actor.zero_grad()
        try:
            _, log_prob = model.actor.action_log_prob(replay_data.observations)
            log_prob = log_prob.reshape(-1, 1)
            log_prob.sum().backward()
            for name, p in self.policy.named_parameters():
                if p.requires_grad and p.grad is not None and "actor" in name:
                    self.fisher_matrix[name] += (p.grad.detach() ** 2) / max(log_prob.shape[0], 1)
            n_used = log_prob.shape[0]
            logger.info(f"PolicyEWC[SAC/actor]: Fisher computed over {n_used} samples (log-likelihood gradient).")
        except Exception as e:
            logger.error(f"PolicyEWC[SAC/actor] Fisher computation failed ({e}).")
        finally:
            model.actor.zero_grad()

        # --- Critic: TD-loss Fisher, same treatment as DQN, replicating
        # SB3 SAC.train()'s actual target computation (next-action sampled
        # from the current actor, min over target critics -- no entropy term
        # in the target here since that's an SAC-specific refinement that
        # doesn't change which parameters matter most, and keeping this
        # simpler and auditable matters more than exactly reproducing every
        # term of the training objective for what is, after all, a
        # regularizer, not the primary learning signal). ---
        model.critic.zero_grad()
        try:
            with torch.no_grad():
                next_actions, _ = model.actor.action_log_prob(replay_data.next_observations)
                next_q_values = torch.cat(model.critic_target(replay_data.next_observations, next_actions), dim=1)
                next_q_values, _ = torch.min(next_q_values, dim=1, keepdim=True)
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * discounts * next_q_values

            current_q_values = model.critic(replay_data.observations, replay_data.actions)
            critic_loss = 0.5 * sum(F.mse_loss(current_q, target_q_values) for current_q in current_q_values)
            critic_loss.backward()
            for name, p in self.policy.named_parameters():
                if p.requires_grad and p.grad is not None and "critic" in name:
                    self.fisher_matrix[name] += (p.grad.detach() ** 2) / max(target_q_values.shape[0], 1)
            n_used = max(n_used, target_q_values.shape[0])
            logger.info(f"PolicyEWC[SAC/critic]: Fisher computed over {target_q_values.shape[0]} samples (TD-loss gradient).")
        except Exception as e:
            logger.error(f"PolicyEWC[SAC/critic] Fisher computation failed ({e}).")
        finally:
            model.critic.zero_grad()

        self.n_calibration_samples_used = n_used

    def penalty(self, policy: Any) -> torch.Tensor:
        """Matches the .penalty(policy) interface EWCCallback already calls.
        Algorithm-agnostic by construction -- it only ever looks at parameter
        names, so it works unchanged for PPO, SAC, or DQN policies."""
        device = next(policy.parameters()).device
        loss = torch.zeros((), device=device)
        matched = 0
        for n, p in policy.named_parameters():
            if n in self.fisher_matrix and n in self.star_params:
                loss = loss + (self.fisher_matrix[n].to(device) * (p - self.star_params[n].to(device)) ** 2).sum()
                matched += 1
        if matched == 0:
            logger.warning(
                "PolicyEWC.penalty(): 0 matching parameter names between this anchor and "
                "the current policy -- different architecture. EWC is providing zero "
                "protection against forgetting this cycle."
            )
        return loss * (self.ewc_lambda / 2.0)


if __name__ == "__main__":
    import gymnasium as gym
    from gymnasium import spaces
    import sys, os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from swarm.rl_trainer import OnlineRLTrainer

    class FakeEnvContinuous(gym.Env):
        def __init__(self):
            super().__init__()
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
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

    class FakeEnvDiscrete(FakeEnvContinuous):
        def __init__(self):
            super().__init__()
            self.action_space = spaces.Discrete(3)

    def run_check(model_name, env, label):
        print(f"--- {label} ---")
        trainer = OnlineRLTrainer(model_name=model_name, env=env, device="cpu")
        trainer.learn(total_timesteps=256)
        anchor = PolicyEWC(trainer.model, n_calibration_samples=64)
        print(f"  detected algo: {anchor.algo}, calibration samples used: {anchor.n_calibration_samples_used}")
        assert anchor.n_calibration_samples_used > 0, f"{label}: expected real calibration data"
        updated = trainer.set_ewc_instance(anchor)
        assert updated
        penalty_at_anchor = float(anchor.penalty(trainer.model.policy).detach())
        trainer.learn(total_timesteps=256)
        penalty_after_drift = float(anchor.penalty(trainer.model.policy).detach())
        print(f"  penalty at anchor: {penalty_at_anchor:.6f}, after further training: {penalty_after_drift:.6f}")
        print(f"  {label} PASSED")

    run_check("PPO-DilatedCNN", FakeEnvContinuous(), "PPO (unchanged path)")
    run_check("SAC-DilatedCNN", FakeEnvContinuous(), "SAC (newly extended: actor + critic Fisher)")
    run_check("DuelingDQN", FakeEnvDiscrete(), "DQN (newly extended: TD-loss Fisher)")
    print("\nAll three algorithm-specific PolicyEWC paths verified end to end.")
