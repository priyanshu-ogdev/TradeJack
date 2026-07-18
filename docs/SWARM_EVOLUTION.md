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

## 2. Dynamic Architecture Swapping (`self_mod_manager.py`)

To guarantee strict compliance with the Warden's VRAM tier assignments, `SelfModEngine` manages three distinct neural architectures inside each container:

1. **`Attention-is-all-you-Need` (Transformer / Multi-Head Attention)**:
   - **VRAM Requirement**: `Tier 1 (20.0 GB)`
   - **Structure**: Multi-head self-attention layers (`MultiheadAttention`), positional encodings, and dense feed-forward networks designed to capture complex long-horizon temporal dependencies across multi-day LOB windows.
2. **`Dilated-CNN-Seq2seq` (Temporal Convolutional Network)**:
   - **VRAM Requirement**: `Tier 2 (4.0 GB)`
   - **Structure**: Stacked 1D dilated causal convolutions (`Conv1d` with exponential dilation factors $1, 2, 4, 8$) that process sequence buffers (`seq_len=60`) with high computational efficiency and low memory overhead.
3. **`Deep-Q-learning` (Scalping Policy Network)**:
   - **VRAM Requirement**: `Tier 3 (0.0 GB / Inference-Only)`
   - **Structure**: Compact multi-layer perceptron (`Linear` -> `ReLU`) optimized for rapid scalping decisions when VRAM is locked or when a container is recovering from an OOM penalty.

If a container's tier changes (`swap_active_architecture`), the engine dynamically instantiates the new topology, pins it to host RAM, and loads surviving parameter embeddings where dimensions match.

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

## 5. Advanced Reinforcement Learning (`rl_mechanics.py`)

- **Hindsight Experience Replay (`HindsightExperienceReplay`)**: When an order execution fails to achieve its target profit (`achieved_equity != desired_equity`), HER modifies the stored transition replay buffer (`sample_with_her`), re-labeling the desired goal as whatever equity outcome was actually achieved. This turns failed trades into informative learning signals.
- **Population-Based Training (`PopulationBasedTrainingEngine`)**: At periodic intervals across the swarm, the bottom $20\%$ of containers (`exploit_fraction=0.4`) discard their underperforming weights, copy the exact parameters of the top $20\%$ containers (`EXPLOIT`), and mutate hyperparameters like learning rate and EWC regularization (`EXPLORE`).
- **Adversarial GAN Spoofing (`AdversarialGANSpoofer`)**: Generates synthetic, adversarial limit order book spoofing patterns (`generate_spoof_pattern`) and injects them during training, ensuring agents learn to detect and ignore fake liquidity imbalances.

---

## 6. P2P Social Relay (`social_relay.py`)

To eliminate centralized control bottlenecks, all containers gossip across a decentralized JSONL message queue (`data_store/relay/messages.jsonl`). Agents broadcast alpha discoveries (`BROADCAST_ALPHA`), VRAM petitions, and hardware warnings (`WARN_OOM`) directly to peers (`P2PMessageRelay`).
