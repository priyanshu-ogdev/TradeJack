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

## 3. Elastic Weight Consolidation (`ewc_optimizer.py`)

Continuous online learning on non-stationary financial data often causes neural networks to suffer from **catastrophic forgetting**—overfitting to the most recent market regime while destroying weights learned during past volatility.

`ElasticWeightConsolidation` (`EWCOptimizer`) prevents this by calculating the diagonal **Fisher Information Matrix ($F$)** across historical calibration batches:

$$\mathcal{L}_{\text{total}}(\theta) = \mathcal{L}_{\text{task}}(\theta) + \sum_i \frac{\lambda}{2} F_i (\theta_i - \theta_{i, \text{star}})^2$$

Where:
- $\theta_{i, \text{star}}$ represents the parameter weights established at the last validated high-water mark checkpoint.
- $F_i$ measures the sensitivity of the loss function to changes in parameter $i$.
- $\lambda = 1000.0$ applies strong regularization, allowing redundant weights to adapt freely while penalizing changes to critical alpha parameters.

---

## 4. Git Financial Rollback (`git_rollback.py`)

Every container tracks its code and weights inside an isolated git repository (`state/child_{id}/self_mod/`).
- **High-Water Mark Tagging (`check_and_checkpoint`)**: Whenever the container attains a new equity high-water mark (`current_equity > hwm_equity`), `GitFinancialRollback` commits all code modifications and weight tensors, creating a permanent git tag (`v_child{id}_{tick}`).
- **Automatic Drawdown Rollback (`execute_rollback_if_breached`)**: If the container enters a severe drawdown exceeding **$15.0\%$**, the engine immediately triggers a hard reset (`git reset --hard <tag>`) back to the last high-water mark tag, immediately stopping the equity bleed and restoring proven strategy code.

---

## 5. Advanced Reinforcement Learning Mechanics (`rl_mechanics.py`)

- **Prioritized Experience Replay Buffer (`PrioritizedReplayBuffer`)**: Off-policy algorithms (SAC, DQN) utilize a centralized, disk-persisted buffer that weights transitions by their TD-error magnitude, ensuring the model samples surprising market conditions more frequently.
- **Hindsight Experience Replay (`sample_with_her`)**: When an order execution fails to achieve its target profit, HER modifies the stored transition replay buffer, re-labeling the desired goal as whatever equity outcome was actually achieved. This turns failed trades into informative learning signals.
- **Population-Based Training (`PopulationBasedTrainingEngine`)**: At periodic intervals across the swarm, the bottom $20\%$ of containers (`exploit_fraction=0.4`) discard their underperforming SB3 `.zip` checkpoints, copy the exact parameters of the top $20\%$ containers (`EXPLOIT`), and mutate hyperparameters like learning rate (`EXPLORE`).
- **Adversarial GAN Spoofing (`AdversarialGANSpoofer`)**: Generates synthetic, adversarial limit order book spoofing patterns and injects them into `LOBEnv` during training, ensuring agents learn to detect and ignore fake liquidity imbalances.

---

## 6. P2P Social Relay (`social_relay.py`)

To eliminate centralized control bottlenecks, all containers gossip across a decentralized JSONL message queue (`data_store/relay/messages.jsonl`). Agents broadcast alpha discoveries (`BROADCAST_ALPHA`), VRAM petitions, and hardware warnings (`WARN_OOM`) directly to peers (`P2PMessageRelay`).
