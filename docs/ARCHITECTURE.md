# Project TradeJack: System Architecture & Design Philosophy

## 1. The Bare-Metal Philosophy vs. Cloud-Native Bots

Modern algorithmic trading bots built for cloud platforms (AWS, GCP, Azure) are plagued by three foundational flaws:
1. **Compute Throttling**: To save server costs, cloud bots sleep between ticks, process order books at low frequency, and rely on pre-computed indicators.
2. **Simulated Sandbox Delusion**: Most quantitative models are trained on idealized historical candles or MT5 demo APIs that assume zero slippage, instant fills, and infinite liquidity.
3. **Monolithic Brittleness**: When a monolithic trading model encounters regime shifts or unforeseen volatility, it either crashes via out-of-memory errors or continues bleeding equity until manual human intervention occurs.

**Project TradeJack** completely rejects this paradigm. Built from the ground up to saturate the **NVIDIA DGX Spark** (Grace Blackwell architecture, 128GB Unified Memory, 1 PetaFLOP compute), TradeJack treats hardware not as a metered expense to be minimized, but as a sovereign, 24/7 evolutionary arena.

---

## 2. Core Architectural Pillars

TradeJack operates as a decentralized, self-replicating swarm of **50 sovereign child containers** governed by a bare-metal hypervisor known as the **Warden**.

```
+-----------------------------------------------------------------------------------+
|                           NVIDIA DGX SPARK HARDWARE                               |
|        (Grace Blackwell CPU/GPU, 128GB Unified Memory, NVLink-C2C, CUDA 13)       |
+-----------------------------------------------------------------------------------+
                                         |
+-----------------------------------------------------------------------------------+
|                        WARDEN BARE-METAL HYPERVISOR                               |
|  - MIG / MVS VRAM Slicing (Tier 1: 20GB | Tier 2: 4GB | Tier 3: 0GB)             |
|  - Logarithmic + Stagnation Survival Tax & Bankruptcy Purging Engine               |
|  - Recklessness Watchdog (-$10 OOM penalty & 24h Tier 3 Lockout)                  |
|  - BlackwellUnifiedAllocator (Zero-copy NVLink memory pinning & swapping)         |
|  - LineageVectorDB (ChromaDB / SQLite Cosine architecture embedding repository)    |
+-----------------------------------------------------------------------------------+
                                         |
            +----------------------------+----------------------------+
            |                            |                            |
+-----------------------+    +-----------------------+    +-----------------------+
|  SOVEREIGN CHILD 0    |    |  SOVEREIGN CHILD 1    |    |  SOVEREIGN CHILD N    |
|                       |    |                       |    |                       |
|  +-----------------+  |    |  +-----------------+  |    |  +-----------------+  |
|  |  SelfModEngine  |  |    |  |  SelfModEngine  |  |    |  |  SelfModEngine  |  |
|  +-----------------+  |    |  +-----------------+  |    |  +-----------------+  |
|  |  EWCOptimizer   |  |    |  |  EWCOptimizer   |  |    |  |  EWCOptimizer   |  |
|  +-----------------+  |    |  +-----------------+  |    |  +-----------------+  |
|  | Git Rollback    |  |    |  | Git Rollback    |  |    |  | Git Rollback    |  |
|  +-----------------+  |    |  +-----------------+  |    |  +-----------------+  |
|  | RL Mechanics    |  |    |  | RL Mechanics    |  |    |  | RL Mechanics    |  |
|  +-----------------+  |    |  +-----------------+  |    |  +-----------------+  |
+-----------------------+    +-----------------------+    +-----------------------+
            |                            |                            |
            +----------------------------+----------------------------+
                                         |
+-----------------------------------------------------------------------------------+
|                   P2P SOCIAL RELAY, ESCROW & 10X AIRGAP                           |
|  - Decentralized Gossip Protocol (`social_relay.py`) across 50 containers          |
|  - Trustless P2PEscrowBridge (Atomic USDC locking & weight ownership transfer)    |
|  - ValidationAirgapEngine (10x out-of-sample historical flash-crash sandboxing)   |
+-----------------------------------------------------------------------------------+
```

---

## 3. The Sovereign Container Lifecycle

Each child container (`child_agent.py`) runs inside an isolated Docker environment (`Dockerfile.child`) equipped with its own SQLite database ledger (`state/child_{id}/ledger.sqlite`) and git tracking repository (`state/child_{id}/self_mod/`).

The container executes an autonomous, continuous loop:
1. **Think & Perceive**: Reads multi-level limit order book depth tensors streamed directly via GPUDirect Storage (`KvikIODataForge`) or DALI loaders (`DALILoader`).
2. **Act & Friction Math**: Emits continuous action vectors to `TradeJackLOBEnv`, where market order sizes are matched against real bid/ask depth tiers, calculating exact slippage, crossing costs, and exchange fees.
3. **Observe & Audit**: Records new equity values in `PortfolioAccountingEngine`. The Warden audits the container, applies the Logarithmic + Stagnation survival tax, and assigns the appropriate VRAM Tier (`Tier 1/2/3`).
4. **Adapt & Evolve**: If the VRAM Tier changes, `SelfModEngine` instantly swaps neural architectures. During online learning, `EWCOptimizer` computes diagonal Fisher Information matrices to prevent catastrophic forgetting.
5. **Protect & Rollback**: If the container's equity drops more than `15.0%` from its historical high-water mark, `GitFinancialRollback` immediately restores code and weights (`git reset --hard`) to the last safe checkpoint.
6. **Relay & Escrow**: The child broadcasts performance heartbeats via `P2PMessageRelay`. If another child discovers superior alpha weights, the buyer initiates a trustless transaction via `P2PEscrowBridge`, which sandboxes the candidate weights across 10 flash-crash splits (`ValidationAirgapEngine`) before releasing USDC payment.

---

## 4. Hardware Independence via Automated Fallbacks

To ensure code can be built, debugged, and tested on standard development laptops without requiring immediate access to the DGX Spark server, all subsystems implement dynamic runtime introspection:
- **PyTorch / CUDA / cuDF**: If PyTorch and CUDA 13 are absent, data ingestion falls back from GPUDirect Storage (`cufile`) to zero-copy Polars/Numpy (`NumpyDataLoader`).
- **ChromaDB**: If ChromaDB vector libraries are not installed locally, `LineageVectorDB` automatically activates a high-speed SQLite and ZSTD-compressed Cosine similarity index.
- **Gymnasium**: If `gymnasium` is not installed, `TradeJackLOBEnv` inherits from a clean internal `BaseEnv` class without breaking API compatibility.
