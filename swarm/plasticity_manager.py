"""
PlasticityManager: periodic, Fisher-guided partial network resets to counter
primacy bias / loss of plasticity during continuous online training.

RESEARCH GROUNDING (not guessed):
  - Primacy bias (Nikishin et al., ICML 2022, "The Primacy Bias in Deep
    Reinforcement Learning"): off-policy RL agents overfit to early
    experience and lose the ability to adapt as more data arrives. The
    established mitigation is periodic resetting of network parameters,
    keeping the replay buffer intact so the freshly-reinitialized network
    relearns from the FULL accumulated data, not just what's arrived since
    the reset.
  - Resetting only the LAST layers, not the whole network, is the
    lower-disruption variant used in follow-up work (D'Oro et al. 2022 "SR-SPR",
    Schwarzer et al. 2023) -- full resets discard the (expensive to relearn)
    feature representation along with the (cheap to relearn, primacy-biased)
    output heads. This file resets only policy/value/Q heads, never the
    shared feature encoder.
  - Selective, Fisher-guided resetting (Fisher-Guided Selective Forgetting,
    2025, arxiv 2502.00802) uses Fisher information to decide WHICH
    parameters to reset -- low-Fisher parameters are less important for
    retaining currently-good behavior, so resetting them is lower-risk than
    resetting everything indiscriminately. This directly reuses the Fisher
    matrices PolicyEWC already computes for a completely different purpose
    (anchoring against forgetting) -- here the same information is used in
    the opposite direction (deciding what's safe to forget on purpose).
  - Resetting the optimizer's momentum state alongside the weights (Asadi et
    al., 2024, "Resetting the Optimizer in Deep RL") -- stale Adam momentum
    computed for the old weights actively fights a freshly-reinitialized
    parameter's early gradients if left in place.
  - Hard-syncing the target network's corresponding head immediately after a
    reset (rather than letting Polyak averaging catch up slowly) avoids a
    period where the target network bootstraps against a completely
    different, non-corresponding value scale than the just-reset online head.

SCOPE, DELIBERATELY NARROW: only resets nn.Linear weight/bias PAIRS found by
name (e.g. "action_net.weight" + "action_net.bias"). Standalone 1D parameters
that aren't part of such a pair (e.g. PPO's `log_std`, a raw exploration-noise
parameter, not a Linear layer) are explicitly skipped and logged rather than
reset with an invented initialization heuristic -- resetting those needs a
different, deliberate treatment, not a guess bolted onto this class.
"""

import math
import logging
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger("PlasticityManager")


# Head parameter name PREFIXES per algorithm, deliberately excluding
# "features_extractor" (the shared encoder) on every branch -- verified
# against the actual named_parameters() output of REGISTRY.build_model() for
# all three algorithms, not assumed from SB3's docs.
_HEAD_PREFIXES = {
    "ppo": ["mlp_extractor.policy_net.", "mlp_extractor.value_net.", "action_net.", "value_net."],
    "sac": ["actor.latent_pi.", "actor.mu.", "critic.qf0.", "critic.qf1."],
    "dqn": ["q_net.q_net."],
}
# Corresponding target-network prefixes to hard-sync after a reset (SAC/DQN
# only -- PPO has no target network).
_TARGET_SYNC_PREFIXES = {
    "sac": [("critic.qf0.", "critic_target.qf0."), ("critic.qf1.", "critic_target.qf1.")],
    "dqn": [("q_net.q_net.", "q_net_target.q_net.")],
}


def _detect_algo(model: Any) -> Optional[str]:
    if hasattr(model, "q_net") and hasattr(model, "q_net_target"):
        return "dqn"
    if hasattr(model, "actor") and hasattr(model, "critic") and hasattr(model, "critic_target"):
        return "sac"
    if hasattr(model, "rollout_buffer"):
        return "ppo"
    return None


def _find_linear_pairs(param_dict: Dict[str, torch.nn.Parameter], prefixes: List[str]) -> List[Tuple[str, str, str]]:
    """Returns (base_name, weight_name, bias_name) triples for every
    weight/bias pair whose name starts with one of `prefixes`. A name
    matching a prefix but missing its bias counterpart (or vice versa) is
    skipped and logged -- resetting a weight without its bias (or the
    reverse) would leave the layer in a state neither the old nor a
    consistent new initialization ever produced."""
    pairs = []
    weight_names = {n for n in param_dict if n.endswith(".weight") and any(n.startswith(p) for p in prefixes)}
    for wn in sorted(weight_names):
        base = wn[: -len(".weight")]
        bn = base + ".bias"
        if bn in param_dict:
            pairs.append((base, wn, bn))
        else:
            logger.warning(f"Found '{wn}' but no matching '{bn}' -- skipping this layer, not resetting only the weight.")
    return pairs


class PlasticityManager:
    """
    Call `maybe_reset(agent_id, model, fisher_matrix)` once per completed
    training cycle (e.g. from ContinuousTrainer, right after a cycle's
    trainer.learn() call, alongside the EWC re-anchoring). Tracks cycles
    elapsed per agent_id independently, since different tournament agents
    may be reset on different schedules if desired.
    """

    def __init__(
        self,
        reset_interval_cycles: int = 5,
        fisher_reset_fraction: float = 0.5,
        use_fisher_guidance: bool = True,
        min_cycles_before_first_reset: int = 2,
    ):
        self.reset_interval_cycles = reset_interval_cycles
        self.fisher_reset_fraction = fisher_reset_fraction
        self.use_fisher_guidance = use_fisher_guidance
        self.min_cycles_before_first_reset = min_cycles_before_first_reset
        self._cycles_since_reset: Dict[str, int] = {}
        self._total_cycles: Dict[str, int] = {}
        self.reset_count = 0
        self.reset_history: List[Dict[str, Any]] = []

    def maybe_reset(self, agent_id: str, model: Any, fisher_matrix: Optional[Dict[str, torch.Tensor]] = None) -> bool:
        self._total_cycles[agent_id] = self._total_cycles.get(agent_id, 0) + 1
        self._cycles_since_reset[agent_id] = self._cycles_since_reset.get(agent_id, 0) + 1

        if self._total_cycles[agent_id] < self.min_cycles_before_first_reset:
            return False
        if self._cycles_since_reset[agent_id] < self.reset_interval_cycles:
            return False

        self._cycles_since_reset[agent_id] = 0
        try:
            reset_names = self._do_reset(model, fisher_matrix)
            self.reset_count += 1
            self.reset_history.append({
                "agent_id": agent_id, "reset_number": self.reset_count,
                "cycle": self._total_cycles[agent_id], "n_layers_reset": len(reset_names),
                "layers": reset_names, "fisher_guided": self.use_fisher_guidance and fisher_matrix is not None,
            })
            logger.warning(
                f"PLASTICITY RESET #{self.reset_count} for agent '{agent_id}' at cycle "
                f"{self._total_cycles[agent_id]}: reset {len(reset_names)} head layer(s): {reset_names}"
            )
            return True
        except Exception as e:
            logger.error(f"Plasticity reset failed for agent '{agent_id}' (non-fatal, training continues): {e}")
            return False

    def _do_reset(self, model: Any, fisher_matrix: Optional[Dict[str, torch.Tensor]]) -> List[str]:
        algo = _detect_algo(model)
        if algo is None:
            logger.warning(f"PlasticityManager: could not detect algorithm on {type(model).__name__} -- nothing reset.")
            return []

        param_dict = dict(model.policy.named_parameters())
        pairs = _find_linear_pairs(param_dict, _HEAD_PREFIXES[algo])
        if not pairs:
            logger.warning(f"PlasticityManager[{algo}]: no head Linear layers found to reset.")
            return []

        target_pairs = pairs
        if self.use_fisher_guidance and fisher_matrix:
            # Rank head layers by mean Fisher importance (lower = less
            # important to retaining current good behavior = safer to
            # reset), and only reset the bottom fraction. If a layer's
            # weight/bias aren't in the Fisher matrix at all (e.g. this is
            # the very first cycle with no EWC anchor yet), treat its
            # importance as 0 (fully eligible for reset) rather than
            # excluding it -- an untracked layer isn't a protected one.
            def _importance(base, wn, bn):
                vals = []
                if wn in fisher_matrix:
                    vals.append(float(fisher_matrix[wn].mean()))
                if bn in fisher_matrix:
                    vals.append(float(fisher_matrix[bn].mean()))
                return sum(vals) / len(vals) if vals else 0.0

            scored = sorted(pairs, key=lambda t: _importance(*t))
            k = max(1, int(round(len(scored) * self.fisher_reset_fraction)))
            target_pairs = scored[:k]
            logger.info(
                f"PlasticityManager[{algo}]: Fisher-guided selection -- resetting {k}/{len(pairs)} "
                f"head layers (lowest importance): {[t[0] for t in target_pairs]}"
            )

        reset_base_names = []
        optimizers_touched = set()
        for base, wn, bn in target_pairs:
            w_param = param_dict[wn]
            b_param = param_dict[bn]
            out_f, in_f = w_param.shape
            fresh_layer = nn.Linear(in_f, out_f)  # real nn.Linear -> guaranteed-correct, standard init, not a hand-rolled formula
            with torch.no_grad():
                w_param.data.copy_(fresh_layer.weight.data.to(w_param.device))
                b_param.data.copy_(fresh_layer.bias.data.to(b_param.device))
            reset_base_names.append(base)

            opt = self._clear_optimizer_state(model, algo, base, w_param, b_param)
            optimizers_touched.update(opt)

        self._sync_target_networks(model, algo, reset_base_names, param_dict)
        return reset_base_names

    def _clear_optimizer_state(self, model: Any, algo: str, base_name: str, w_param, b_param) -> List[str]:
        """Clears Adam's exp_avg/exp_avg_sq/step for the just-reset
        parameters so stale momentum computed for the OLD weights doesn't
        fight the fresh initialization's first few gradients."""
        touched = []
        if algo == "sac":
            optimizer = model.critic.optimizer if base_name.startswith("critic.") else model.actor.optimizer
            touched.append("critic" if base_name.startswith("critic.") else "actor")
        else:
            optimizer = getattr(model.policy, "optimizer", None)
            touched.append("policy")
        if optimizer is None:
            return []
        for p in (w_param, b_param):
            if p in optimizer.state:
                del optimizer.state[p]
        return touched

    def _sync_target_networks(self, model: Any, algo: str, reset_base_names: List[str], param_dict: Dict[str, Any]):
        """Hard-copies the just-reset online head directly into the
        corresponding target-network head, instead of letting Polyak
        averaging (tau ~0.005) catch up over hundreds of steps -- during
        that catch-up window the target would otherwise bootstrap Q-value
        estimates against a head that no longer corresponds to anything the
        online network still produces."""
        if algo not in _TARGET_SYNC_PREFIXES:
            return
        for online_prefix, target_prefix in _TARGET_SYNC_PREFIXES[algo]:
            base_match = [b for b in reset_base_names if b.startswith(online_prefix)]
            if not base_match:
                continue
            for base in base_match:
                target_base = target_prefix + base[len(online_prefix):]
                w_name, b_name = base + ".weight", base + ".bias"
                tw_name, tb_name = target_base + ".weight", target_base + ".bias"
                if tw_name in param_dict and tb_name in param_dict:
                    with torch.no_grad():
                        param_dict[tw_name].data.copy_(param_dict[w_name].data)
                        param_dict[tb_name].data.copy_(param_dict[b_name].data)
                    logger.info(f"Hard-synced target head '{tw_name}'/'{tb_name}' to match reset online head.")


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    import numpy as np
    import gymnasium as gym
    from gymnasium import spaces
    from swarm.rl_trainer import OnlineRLTrainer
    from swarm.ewc_sb3_adapter import PolicyEWC

    class FakeEnvC(gym.Env):
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
            return {"lob_sequence": np.random.randn(64,5).astype(np.float32), "portfolio_state": np.random.randn(4).astype(np.float32)}, {}
        def step(self, a):
            self.t += 1
            obs = {"lob_sequence": np.random.randn(64,5).astype(np.float32), "portfolio_state": np.random.randn(4).astype(np.float32)}
            return obs, float(np.random.randn()*0.01), self.t >= 50, False, {}

    class FakeEnvD(FakeEnvC):
        def __init__(self):
            super().__init__()
            self.action_space = spaces.Discrete(3)

    def check(model_name, env, label):
        print(f"--- {label} ---")
        trainer = OnlineRLTrainer(model_name=model_name, env=env, device="cpu")
        trainer.learn(total_timesteps=128)
        anchor = PolicyEWC(trainer.model, n_calibration_samples=64)

        pm = PlasticityManager(reset_interval_cycles=1, min_cycles_before_first_reset=1, fisher_reset_fraction=0.5)
        head_names = [n for pfx in _HEAD_PREFIXES[_detect_algo(trainer.model)] for n in dict(trainer.model.policy.named_parameters()) if n.startswith(pfx)]
        before = {n: p.clone().detach() for n, p in trainer.model.policy.named_parameters() if n in head_names}
        encoder_before = {n: p.clone().detach() for n, p in trainer.model.policy.named_parameters() if "features_extractor" in n}

        did_reset = pm.maybe_reset("test_agent", trainer.model, fisher_matrix=anchor.fisher_matrix)
        assert did_reset, "expected a reset on the very first eligible cycle"

        after = dict(trainer.model.policy.named_parameters())
        n_changed = sum(1 for n, p in before.items() if not torch.allclose(p, after[n]))
        n_encoder_changed = sum(1 for n, p in encoder_before.items() if not torch.allclose(p, after[n]))
        print(f"  head params changed: {n_changed}/{len(before)}  (expect > 0, and NOT necessarily all -- Fisher-guided)")
        print(f"  encoder params changed: {n_encoder_changed}/{len(encoder_before)}  (must be exactly 0 -- encoder must never be touched)")
        assert n_changed > 0, "expected at least some head parameters to actually change"
        assert n_encoder_changed == 0, "PLASTICITY RESET TOUCHED THE SHARED ENCODER -- this must never happen"

        # confirm training still works after a reset (no crash, no shape mismatch)
        trainer.learn(total_timesteps=64)
        print(f"  training after reset: OK")
        print(f"  {label} PASSED\n")

    check("PPO-DilatedCNN", FakeEnvC(), "PPO")
    check("SAC-DilatedCNN", FakeEnvC(), "SAC")
    check("DuelingDQN", FakeEnvD(), "DQN")
    print("All PlasticityManager checks passed.")
