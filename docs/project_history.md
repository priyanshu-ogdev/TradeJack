# Project TradeJack: Comprehensive Development History & Roadmap

This document serves as the master chronological log of all architectural decisions, implementations, and verified milestones for **Project TradeJack**.

---

## 🏛️ 1. Project Genesis & Core Constraints
**The Vision:** Build a completely sovereign, self-contained AI trading swarm that evolves dynamically without human intervention.
**The Inspirations:** Fusing the autonomous survival, self-modification, and P2P social relay mechanics from `Conway-Research/automaton` with the advanced predictive neural architectures from `huseinzol05/Stock-Prediction-Models`.
**The Hardware Target:** Uncompromisingly designed for a local bare-metal **NVIDIA DGX Spark** (1 PetaFLOP, 128GB Unified Memory, Grace Blackwell, CUDA 13). No cloud costs, no MT5 demo accounts—just 100% hardware saturation 24/7.
**The Survival Physics:** Each swarm agent (Child) spawns with exactly $10.00. They face a relentless logarithmic stagnation tax ($- \alpha \ln(1+t) - \beta t_{\text{stag}}$) and must trade profitably across a synthetic Limit Order Book (LOB) to survive.

---

## 🏗️ 2. Phase 1: Environment & Physics Engine (Completed)
We first built the underlying reality that the swarm agents live in.
- **KvikIODataForge (`data_forge/`)**: Built a high-throughput, zero-copy synthetic LOB data pipeline capable of generating realistically noisy order book depth states (150+ ticks/day). Fully fallback-compatible with Numpy/CPU when CUDA is unavailable during laptop development.
- **TradeJackLOBEnv (`physics/lob_env.py`)**: A vectorized, high-performance Gymnasium-style environment mirroring real-world slippage, latency, and maker/taker fees. 
- **Portfolio Accounting (`physics/portfolio_tracker.py`)**: Strict immutable ledger to track live equity, max drawdown, Sharpe, and Sortino ratios for every container.

---

## 🧠 3. Phase 2: Hardware Hypervisor & Warden (Completed)
To prevent the 50 Docker containers from crashing the GPU via Out-Of-Memory (OOM) errors, we built the ultimate hardware gatekeeper.
- **Warden Core (`warden/warden_core.py`)**: Implemented NVIDIA MIG (Multi-Instance GPU) partitioning logic to enforce strict VRAM memory tiers across containers.
- **Memory Tiers**: 
  - **Tier 1 (High Alpha)**: 20GB VRAM allowed (for complex sequence models).
  - **Tier 2 (Standard Compute)**: 4GB VRAM allowed.
  - **Tier 3 (Survival Mode)**: 1GB / Inference-only / FP8 allowed.
- **OOM Watchdogs & Penalties**: Hardware watchdogs that instantly detect CUDA OOM, enforce a $10 financial penalty, and forcefully drop the violating container down to Tier 3.

---

## 🔐 4. Phase 3: P2P Swarm Network & Trustless Escrow (Completed)
We implemented the social relay and security mechanisms for autonomous knowledge sharing.
- **Social Relay Bridge (`swarm/social_relay.py`)**: Built a P2P gossip protocol where containers can advertise their successful weights (lineages) using cryptographically secured HMAC-SHA256 signatures, random nonces, and ULID packet IDs to prevent replay attacks.
- **Trustless Escrow (`escrow/escrow_contract.py`)**: When a struggling agent wants to buy weights from a successful peer, funds (e.g., \$2.00 USDC) are temporarily locked in `escrow_ledger.sqlite`.
- **10x Validation Airgap (`escrow/validation_airgap.py`)**: Before funds settle and weights are transferred, the Warden sandboxes the weights and forces them through 10 out-of-sample stress data splits. If the model is poisoned or overfitted (Sharpe < 1.0 or Drawdown > 15%), the transaction is instantly rejected and the buyer is 100% refunded.

---

## 🧬 5. Phase 4: Universal Model Registry & Neuro-Evolution (Completed)
We fused the actual neural architectures into the swarm container logic.
- **Universal Model Registry (`swarm/model_registry.py`)**: Implemented 21+ neural architectures (Transformers, CNNs, VAEs, Actor-Critic, Q-Learning, Neuro-Evolution, Stacking Ensembles) categorized strictly into the 3 hardware tiers.
- **Dynamic Self-Modification (`swarm/self_mod_manager.py`)**: Containers dynamically detect their Warden VRAM tier and equity state (`HIGH`, `NORMAL`, `LOW_COMPUTE`, `CRITICAL`). If an agent bleeds cash and hits `CRITICAL`, the `SelfModEngine` automatically downgrades its neural architecture (e.g., from a heavy Transformer to a lightweight `Curiosity-Q-learning-Agent`) to minimize compute tax and survive.
- **PBT Architecture Mutation (`swarm/rl_mechanics.py`)**: During Population-Based Training, failing agents do not just copy weights; they actively dump their entire neural architecture (`model_name`) to inherit the successful parent's architecture, driving true macro-evolution across the swarm.
- **Git Rollbacks**: Automatic HWM (High-Water Mark) checkpoints and instant codebase rollbacks if a container suffers a drawdown >15%.

---

## 🚀 6. Phase 5: Genesis Orchestration & SOTA Hardening (Completed)
We successfully integrated the Data Forge, Physics Engine, Warden, and Swarm into the master `genesis_prime.py` orchestrator and sealed the "5 Silent Killers" of bare-metal execution:
- **The Reaper Protocol**: OS-level signal handlers (`SIGINT`, `SIGTERM`, `atexit`) intercept aborts and violently tear down the vLLM Shared Brain and Warden processes, preventing Zombie containers from holding VRAM hostage.
- **Strict Environment Masking**: The master process blanks `CUDA_VISIBLE_DEVICES` after hardware validation, physically restricting PyTorch context fragmentation on the DGX Grace Blackwell Unified Memory pool.
- **Aggressive CUDA Purging**: During `SelfModEngine` model swaps, the agent actively pushes weights to CPU, executes `gc.collect()`, deletes the python `sys.modules` reference, and sweeps `torch.cuda.empty_cache()` to permanently eliminate memory creep.
- **Multiprocessing SQLite Audit Isolation**: The Warden's tax assessment loop (`run_audit_cycle`) was pulled out of Python's Thread GIL and injected into an isolated `multiprocessing.Process`. This guarantees zero micro-stutters on the FastAPI network when executing heavy SQL joins across 50 agent databases simultaneously.
- **Idempotent Taxation**: SQLite schemas in `portfolio_tracker.py` and `warden_core.py` now enforce `UNIQUE(child_id, market_timestamp, tax_type)` with `INSERT OR IGNORE` logic, making asynchronous double-dip taxation mathematically impossible.

**Final Status**: Systems Nominal. Project TradeJack is sealed, verified, and fundamentally immortal on bare-metal execution.
