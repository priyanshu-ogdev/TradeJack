# Sovereign Swarm Automaton & Neuro-Evolution

The `swarm/` package contains the sovereign execution logic (`child_agent.py`), self-modifying architecture manager, online learning drift protection, and decentralized social gossip protocols.

---

## 1. Sovereign Child Main Loop (`child_agent.py`)

Each container executes `SovereignChild.run_crucible_loop()`, an autonomous, continuous loop:
```
[Think: Stream LOB Tensor] -> [Act: Execute Order in LOBEnv] -> [Observe: Record Ledger & Check Drawdown]
          ^                                                                             |
          +---- [Adapt: Swap Model Tier | Check EWC Fisher | Relay Gossip | Rollback] <-+
```

---

## 2. Standardized Architecture Swapping (`stable-baselines3` & `model_registry.py`)

To guarantee strict compliance with the Warden's VRAM tier assignments and utilize production-grade reinforcement learning, `model_registry.py` manages three distinct neural architectures inside each container, wrapped natively by `stable-baselines3`:

1. **`PPO-DilatedCNN` (Proximal Policy Optimization)**:
   - **VRAM Requirement**: `Tier 1 (20.0 GB)` / High compute
   - **Structure**: On-policy advantage actor-critic. Uses the shared `LOBFeatureEncoder` (dilated causal CNN, features_dim=128) to capture complex long-horizon temporal dependencies across multi-day LOB windows.
2. **`SAC-DilatedCNN` (Soft Actor-Critic)**:
   - **VRAM Requirement**: `Tier 2 (4.0 GB)` / Medium compute
   - **Structure**: Off-policy maximum entropy RL. Uses the same `LOBFeatureEncoder` but relies on the `PrioritizedReplayBuffer` for highly sample-efficient exploration.
3. **`DuelingDQN` (Deep Q-Network)**:
   - **VRAM Requirement**: `Tier 3 (1.0 GB / Inference-Optimized)`
   - **Structure**: Compact off-policy network for discrete value-based scalping decisions when VRAM is locked or when a container is recovering from an OOM penalty.

All architectures share the `LOBFeatureEncoder`/`LOBFeatureExtractor` which collapses LOB sequence variables and parallel portfolio states into a unified latent space.

### Phase 5 Upgrades: The Ghost Brain & Memory Creep Fixes
In Phase 5, the `SelfModEngine` received critical hardening to survive 100-day hot-swapping without leaking VRAM:
- **Aggressive CUDA Purge**: Before swapping architectures, the engine pushes the old model's weights to CPU (`old_model.cpu()`), deletes the instance, executes `gc.collect()`, deletes the module from Python's C++ cache (`del sys.modules["active_model"]`), and sweeps the physical GPU registers with `torch.cuda.empty_cache()`.
- **Dynamic AST Module Re-loading**: When the agent requests a code-rewrite from the Automaton reasoning bridge, the new python file is loaded dynamically via `importlib.util.spec_from_file_location`, bypassing the Ghost Brain bug where Python refused to clear the old AST structure.
- **Sandboxed Dry-Runs**: Rewritten code is executed in an isolated `subprocess.run` with a strict `timeout=15.0`. If the LLM generated an infinite `while` loop or malformed tensor shapes, the subprocess crashes safely, and the agent rolls back without corrupting the live trading loop.

---

## 3. Elastic Weight Consolidation — now actually wired for SB3 (`swarm/ewc_sb3_adapter.py`)

Continuous online learning on non-stationary financial data often causes neural networks to suffer from **catastrophic forgetting**—overfitting to the most recent market regime while destroying weights learned during past volatility.

**Correction to this section, based on tracing actual execution rather than reading class names**: `ewc_optimizer.py`'s `ElasticWeightConsolidation` — described below for historical reference — assumes a plain regression model (`outputs = self.model(batch_x)` scored against an external label `batch_y`). SB3 policies don't have that shape (`policy.forward(obs)` returns actions/values/log-probs, not a single label-comparable tensor), and in the v3 SB3-based pipeline this class was never actually constructed anywhere except a unit test and one dead code branch in `self_mod_manager.py` gated on a `calibration_loader` argument nothing ever supplies — so despite `EWCCallback` existing in `model_registry.py`, EWC provided **zero actual protection against forgetting** until this was fixed.

The real, verified implementation is `swarm/ewc_sb3_adapter.py`'s `PolicyEWC`, which computes the Fisher information per algorithm using each algorithm's own actual training loss (not swapped in ad hoc — replicated from SB3's own `train()` source):

- **PPO**: Fisher = squared gradient of `log pi(a|s)` for actions actually taken, sampled from the model's own rollout buffer. This is the textbook EWC definition (Kirkpatrick et al., 2017) applied directly.
- **DQN**: Kirkpatrick et al.'s own paper applies EWC to DQN on Atari using the TD/Huber loss gradient in place of a log-likelihood (a deterministic argmax policy has no likelihood to take). This file replicates SB3 `DQN.train()`'s exact target-network computation so the Fisher signal matches what the optimizer actually minimizes.
- **SAC**: Following Powers et al. (2021, "Same State, Different Task"), Fisher is computed on **both** the actor (log-likelihood, as in PPO) and the critic (TD-loss, as in DQN), in two separate backward passes.

$$\mathcal{L}_{\text{total}}(\theta) = \mathcal{L}_{\text{task}}(\theta) + \sum_i \frac{\lambda}{2} F_i (\theta_i - \theta_{i, \text{star}})^2$$

`PolicyEWC` is constructed fresh after every promotion in `training/continuous_trainer.py._promote_champion()`, anchored to the just-promoted champion, and propagated to every tournament agent via `OnlineRLTrainer.set_ewc_instance()` — so the *next* training cycle is regularized against forgetting what just got promoted. A second bug was found and fixed in the same pass: the correction step originally assumed a single `policy.optimizer`, true for PPO/DQN but not SAC (which has separate `actor.optimizer`/`critic.optimizer`) — this silently no-opped every SAC cycle until fixed to detect and step whichever optimizer layout the algorithm actually has.

**Known scope boundary, not silently glossed over**: `PolicyEWC.penalty()` only protects parameters it has Fisher information for — an agent whose architecture doesn't match the anchor's (e.g. a DQN agent being handed a PPO-derived anchor) correctly gets zero protection and logs a warning, rather than silently doing nothing or crashing.

---

## 3a. Primacy Bias / Plasticity Loss Mitigation (`swarm/plasticity_manager.py`)

A system training forever on live, non-stationary market data is close to exactly the setting studied by continual-RL "loss of plasticity" and "primacy bias" research — off-policy agents can overfit to early experience and lose the ability to adapt as more (and different-regime) data arrives (Nikishin et al., ICML 2022).

`PlasticityManager` periodically resets only the policy/value/Q **head** layers (never the shared `LOBFeatureEncoder`, verified by checking zero encoder parameters change on every reset), using a real `nn.Linear`'s own initialization rather than a hand-rolled formula. When a `PolicyEWC` anchor is available from the current cycle, the reset is **Fisher-guided** (arxiv 2502.00802): only the lowest-importance fraction of head layers are reset, reusing the same Fisher matrices EWC already computes for the opposite purpose (deciding what's safe to forget, instead of what to protect). Optimizer momentum state is cleared for reset parameters (Asadi et al., 2024), and SAC/DQN target networks are hard-synced to their freshly-reset online counterparts immediately, rather than left to catch up slowly via Polyak averaging against a head that no longer corresponds to anything.

Wired into `ContinuousTrainer`'s training loop on its own schedule (`plasticity_reset_interval_cycles`, default every 5 cycles), independent of the EWC re-anchoring cadence — primacy bias accumulates with training steps regardless of whether a promotion happened recently.

**Honest limitation**: the reset interval is a literature-informed default, not tuned against this project's actual data, and there is no automated comparison yet proving resets *improve* long-run performance here versus not resetting — the mechanism is verified to work correctly (encoder untouched, training continues cleanly afterward, target networks stay consistent), not yet verified to help on this specific task.

---

## 4. Git Financial Rollback (`git_rollback.py`)

Every container tracks its code and weights inside an isolated git repository (`state/child_{id}/self_mod/`).
- **High-Water Mark Tagging (`check_and_checkpoint`)**: Whenever the container attains a new equity high-water mark (`current_equity > hwm_equity`), `GitFinancialRollback` commits all code modifications and weight tensors, creating a permanent git tag (`v_child{id}_{tick}`).
- **Automatic Drawdown Rollback (`execute_rollback_if_breached`)**: If the container enters a severe drawdown exceeding **$15.0\%$**, the engine immediately triggers a hard reset (`git reset --hard <tag>`) back to the last high-water mark tag, immediately stopping the equity bleed and restoring proven strategy code.

---

## 5. Advanced Reinforcement Learning Mechanics (`rl_mechanics.py`, `swarm/sb3_replay_buffer_adapter.py`)

- **Prioritized Experience Replay + Hindsight Experience Replay, now actually wired (`SB3ReplayBufferAdapter`)**: `swarm/replay_buffer.py`'s `PrioritizedReplayBuffer` and its `sample_with_her` HER re-labeling existed in earlier versions of this codebase but were never connected to anything SB3 actually calls — `sample_with_her()` was confirmed to have zero callers outside its own unit test and `__main__` block, meaning SAC/DQN trained against SB3's default uniform-sampling buffer regardless of this class's existence. `swarm/sb3_replay_buffer_adapter.py`'s `SB3ReplayBufferAdapter` is the actual translation layer this needed — it satisfies SB3's exact `add()`/`sample()`/`DictReplayBufferSamples` contract while delegating storage to `PrioritizedReplayBuffer`, and is now passed as `replay_buffer_class` for SAC and DQN in `model_registry.py` by default. Verified through the real `REGISTRY.build_model()` path (not a standalone reimplementation) that both algorithms build with this buffer and produce real gradient updates sampling from it.
- **Population-Based Training (`PopulationBasedTrainingEngine`)**: At periodic intervals across the swarm, the bottom $20\%$ of containers (`exploit_fraction=0.4`) discard their underperforming SB3 `.zip` checkpoints, copy the exact parameters of the top $20\%$ containers (`EXPLOIT`), and mutate hyperparameters like learning rate (`EXPLORE`).
- **Adversarial GAN Spoofing (`AdversarialGANSpoofer`)**: Generates synthetic, adversarial limit order book spoofing patterns and injects them into `LOBEnv` during training, ensuring agents learn to detect and ignore fake liquidity imbalances.

---

## 6. P2P Social Relay (`social_relay.py`)

To eliminate centralized control bottlenecks, all containers gossip across a decentralized JSONL message queue (`data_store/relay/messages.jsonl`). Agents broadcast alpha discoveries (`BROADCAST_ALPHA`), VRAM petitions, and hardware warnings (`WARN_OOM`) directly to peers (`P2PMessageRelay`).
