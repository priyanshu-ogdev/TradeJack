# Warden Hypervisor & Memory Controller

The `warden/` package is the bare-metal hypervisor responsible for managing hardware boundaries, enforcing economic survival rules, preventing CUDA out-of-memory crashes, and tracking evolutionary lineage across the 50-container swarm.

---

## 1. Dynamic MIG / MVS VRAM Partitioning (`warden_core.py`)

On Grace Blackwell, the `WardenHypervisor` dynamically slices physical hardware capabilities based on real-time financial performance (`audit_child_ledger`). At every audit cycle, each container is assigned to one of three strict hardware tiers:

| Tier | Name | VRAM Allocation | Eligibility Thresholds | Purpose |
| :---: | :--- | :---: | :--- | :--- |
| **Tier 1** | **High Alpha** | `20.0 GB VRAM` | Sharpe > `2.0`, Drawdown < `10%`, Equity >= `$12.00` | Full multi-head attention (`Attention-is-all-you-Need`) & rapid online transformer finetuning. |
| **Tier 2** | **Stagnant / Mid** | `4.0 GB VRAM` | Sharpe > `0.5`, Drawdown < `25%`, or New Child (`ticks_active < 10`) | Mid-sized convolutional sequence encoders (`Dilated-CNN-Seq2seq`). |
| **Tier 3** | **Failing / OOM Lock** | `0.0 GB VRAM` (Inference-Only) | Sharpe <= `0.5`, Drawdown >= `25%`, or Active OOM Lockout | Lightweight scalping heuristics (`Deep-Q-learning`) executed via CPU or zero-copy inference. |

---

## 2. Logarithmic + Stagnation Survival Tax (`apply_survival_tax`)

To prevent containers from hoarding capital or idling ("turtling"), the Warden enforces a continuous burn rate deducted directly from each child's SQLite ledger (`state/child_{id}/ledger.sqlite`):

$$\text{Tax}_t = \text{Tax}_0 \cdot \left(1 + \alpha \cdot \ln\left(1 + \frac{t}{60}\right)\right) + \beta \cdot \max(0, t_{\text{stagnant}} - 10)$$

Where:
- $\text{Tax}_0 = \$1.00 / \text{hr}$ base survival cost.
- $\alpha = 0.5$ tax scale increasing logarithmically with container age ($t$ in simulated minutes).
- $\beta = \$0.10 / \text{tick}$ penalty levied whenever a container remains stagnant without generating positive equity growth for more than 10 ticks.

If a container's equity drops below $\$0.00$ or cash drops below $-\$50.00$, the Warden immediately calls `terminate_insolvent_child()`, stopping the Docker container, archiving its logs, and purging its ledger.

---

## 3. Zero-OOM Unified Memory Swapping (`unified_memory_swap.py`)

The **Grace Blackwell architecture** connects the Grace CPU and Blackwell GPU via high-speed **NVLink-C2C**, creating a single **128GB Unified Memory** address space. `BlackwellUnifiedAllocator` exploits this physical capability to completely eliminate CUDA out-of-memory crashes:

- **Host Pinning (`pin_to_unified_memory`)**: Candidate model weights and dormant state dictionaries are pinned in host RAM using `t_host.pin_memory()`. Because NVLink-C2C allows direct GPU read access to pinned host memory at up to 900 GB/s, models can be evaluated or swapped into active CUDA execution units (`swap_into_active_cuda`) in microseconds.
- **VRAM Quota Management**: Sets exact per-process CUDA memory limits (`torch.cuda.set_per_process_memory_fraction`), ensuring no single container can monopolize physical GPU registers.

---

## 4. Recklessness Watchdog (`oom_watchdog.py`)

The `RecklessnessWatchdog` continuously monitors container execution for memory violations and crashes:
- If a child attempts to allocate memory beyond its assigned MIG quota and triggers a `CUDA_OOM_KILLED` or `MemoryError`, the watchdog intercepts the event.
- It immediately docks a **$-\$10.00$ penalty** directly from the offending container's cash reserve.
- It sets an `oom_penalty_active = True` lock and enforces a mandatory **24-hour Tier 3 (Inference-Only) Lockout** (`oom_lock_until = time.time() + 86400`).

---

## 5. Evolutionary Lineage Repository (`lineage_vector_db.py`)

When a container achieves superior alpha or discovers a breakthrough neural architecture, the Warden records the model topology and its flattened parameter embeddings inside `LineageVectorDB`.
- Uses **ChromaDB** when installed, or falls back to a custom **SQLite + ZSTD Cosine Similarity Index** on local CPUs.
- Enables new child containers or surviving agents to query top historical architectures (`query_surviving_weights`), preventing the swarm from re-discovering dead or poisoned weight configurations.
