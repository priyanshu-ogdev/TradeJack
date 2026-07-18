# Project TradeJack: Sovereign Bare-Metal AI Swarm Architecture

[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![NVIDIA CUDA 13](https://img.shields.io/badge/CUDA-13.0-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![Grace Blackwell](https://img.shields.io/badge/Hardware-DGX%20Spark%20%28Blackwell%29-76B900.svg)](https://www.nvidia.com/en-us/data-center/dgx-platform/)
[![Test Suite](https://img.shields.io/badge/Tests-16%20Passed%20%28100%25%29-brightgreen.svg)](tests/)

**Project TradeJack** is an uncompromising, bare-metal, self-replicating neuro-evolutionary financial organism engineered specifically for the **NVIDIA DGX Spark** (Grace Blackwell architecture, CUDA 13, 128GB Unified Memory, 1 PetaFLOP compute). It fuses cutting-edge mathematical perception models (`huseinzol05/Stock-Prediction-Models`) with sovereign, self-modifying agentic execution (`Conway-Research/automaton`).

Unlike traditional cloud-native bots that throttle compute or rely on simulated MT5 demo APIs, TradeJack operates as a 24/7 evolutionary crucible across **50 sovereign Docker containers**. Each container governs its own economic survival, competes for hardware VRAM slices under real-time survival taxation, evolves neural topologies, checks for catastrophic forgetting, and engages in trustless peer-to-peer (P2P) weight exchange via smart contract escrow and 10x out-of-sample validation airgaps.

---

## 🚀 Key Features & Subsystems

1. **Bare-Metal Warden Hypervisor & Memory Controller (`warden/`)**
   - **MIG/MVS Dynamic VRAM Partitioning**: Slices the Blackwell GPU across containers into **Tier 1 (20GB / High Priority)**, **Tier 2 (4GB / Mid Priority)**, or **Tier 3 (0GB / Inference-Only)** based on live economic metrics (Sharpe ratio and max drawdown).
   - **Logarithmic + Stagnation Survival Tax**: Automatically deducts continuous economic rent (`Tax_t = Tax_0 * (1 + alpha * ln(1 + t/60)) + beta * t_stagnant`) from each child's SQLite ledger. Insolvent containers (`equity <= $0` or `cash <= -$50`) are terminated and purged.
   - **Zero-OOM Unified Memory Swapping**: Exploits Grace Blackwell's 128GB Unified Memory pool sharing physical RAM between CPU and GPU over NVLink-C2C (`BlackwellUnifiedAllocator`). Models are pinned in host memory and swapped into active CUDA execution units in microseconds without CUDA Out-Of-Memory crashes.
   - **Recklessness Watchdog**: Monitors Docker events and memory allocations. Any container triggering a CUDA OOM is instantly penalized `-$10.00` and locked to Tier 3 for 24 hours.

2. **High-Throughput Data Forge (`data_forge/`)**
   - **GPUDirect Storage (`cufile`) Streaming**: `KvikIODataForge` streams partitioned limit order book data (`data_store/symbol/year/month/day/depth.parquet` or `depth.npz`) directly from NVME drives into CUDA tensor memory without CPU bounce buffers.
   - **Zero-Copy DALI & Polars Pipelines**: Utilizes NVIDIA DALI batch loaders with double-buffer sequence batching on hardware, or clean Polars/Numpy binary fallbacks on local CPU laptops.

3. **Friction-Injected LOB Physics Engine (`physics/`)**
   - **Exact Order Book Slippage (`TradeJackLOBEnv`)**: No simplified fill assumptions. Market orders (`compute_market_order_fill`) traverse 8 exact limit order book bid/ask tiers, calculating volume-weighted average fill prices, bid-ask spread crossing costs, and transaction fees.
   - **Real-Time Portfolio Accounting**: `PortfolioAccountingEngine` maintains per-container SQLite ledgers (`state/child_{id}/ledger.sqlite`), computing continuous Sharpe ratio, Sortino ratio, high-water mark equity, and maximum drawdown.

4. **Sovereign Swarm Automaton & Neuro-Evolution (`swarm/`)**
   - **Self-Modifying Architecture Engine (`SelfModEngine`)**: Dynamically swaps neural architectures (`Attention-is-all-you-Need` Transformer, `Dilated-CNN-Seq2seq`, or `Deep-Q-learning` scalper) to strictly comply with the assigned VRAM Tier memory limits.
   - **Elastic Weight Consolidation (`EWCOptimizer`)**: Calculates diagonal Fisher Information matrices across historical calibration datasets to penalize drift on critical alpha weights, eliminating catastrophic forgetting during continuous online adaptation.
   - **Git Financial Rollback (`GitFinancialRollback`)**: Automatically tags repository state at equity high-water marks (`v_child{id}_{tick}`). If a container experiences a drawdown exceeding `15.0%`, the system instantly resets code commits and weights back to the exact high-water mark tag (`git reset --hard`).
   - **Advanced Reinforcement Learning (`RLMechanics`)**: Integrates **Hindsight Experience Replay (HER)** to convert failed execution sequences into positive training targets, **Population-Based Training (PBT)** where bottom 20% containers exploit and mutate top 20% parent hyperparameters, and **Adversarial GAN Spoofing** to harden models against fake LOB imbalances.

5. **Trustless P2P Escrow & 10x Validation Airgap (`escrow/`)**
   - **Atomic P2P Escrow Bridge (`P2PEscrowBridge`)**: Enables containers to discover and purchase superior model weights from peer containers via decentralized social gossip (`social_relay.py`). Atomically locks the agreed USDC cash balance during validation.
   - **10x Out-of-Sample Validation Airgap (`ValidationAirgapEngine`)**: Sandboxes candidate model weights across 10 severe historical flash-crash splits (`2020-03-12`, `2021-05-19`, etc.). If the candidate fails to maintain `Avg Sharpe >= 1.0` and `Max Drawdown <= 15.0%`, the transaction is rejected and the buyer receives an automatic **100% refund (`REFUNDED`)**.

---

## 📁 Repository Structure

```
d:\TradeJack\
├── warden/                       # Component 1: Hypervisor, MIG/MVS Controller & Watchdogs
│   ├── __init__.py
│   ├── warden_core.py            # WardenHypervisor, MIG slicing, Logarithmic/Stagnation Tax
│   ├── unified_memory_swap.py    # BlackwellUnifiedAllocator (128GB Unified Memory NVLink-C2C)
│   ├── oom_watchdog.py           # RecklessnessWatchdog (-$10 penalty & 24h Tier 3 lock)
│   ├── lineage_vector_db.py      # ChromaDB / SQLite Cosine vector lineage tracking
│   ├── vllm_server.py            # Local vLLM reasoning server for Tier 1 containers
│   └── compute_server.py         # Local HTTP Server (`/petition_vram`, `/report_heartbeat`)
├── data_forge/                   # Component 2: High-Performance Data Streaming
│   ├── __init__.py
│   ├── kvikio_pipeline.py        # KvikIO GPUDirect Storage tensor streaming & Polars/NPZ loader
│   ├── parquet_ingest.py         # Daily LOB depth partitioner & synthetic data generator
│   └── dali_loader.py            # NVIDIA DALI zero-copy batch iterators (`create_lob_dataloader`)
├── physics/                      # Component 3: LOB Execution Physics & Ledger
│   ├── __init__.py
│   ├── lob_env.py                # TradeJackLOBEnv exact slippage & spread deduction
│   └── portfolio_tracker.py      # PortfolioAccountingEngine SQLite state & drawdown metrics
├── swarm/                        # Component 4: Swarm Automaton & Neuro-Evolution
│   ├── __init__.py
│   ├── child_agent.py            # SovereignChild main Think->Act->Observe->Adapt loop
│   ├── self_mod_manager.py       # SelfModEngine dynamic memory-bound architecture swapping
│   ├── ewc_optimizer.py          # ElasticWeightConsolidation diagonal Fisher drift prevention
│   ├── git_rollback.py           # GitFinancialRollback high-water mark tags & 15% rollback
│   ├── rl_mechanics.py           # Hindsight Experience Replay, PBT, & Adversarial GAN spoofing
│   └── social_relay.py           # Decentralized P2P message gossip protocol (`messages.jsonl`)
├── escrow/                       # Component 5: P2P Escrow & Stress Airgap
│   ├── __init__.py
│   ├── escrow_contract.py        # P2PEscrowBridge trustless USDC locking & settlement
│   └── validation_airgap.py      # ValidationAirgapEngine 10x historical flash-crash sandboxing
├── scripts/                      # Component 6: Orchestration & Production Deployment
│   ├── Dockerfile.child          # Production container build (CUDA 13, cuDF, KvikIO, PyTorch)
│   └── genesis_prime.py          # Master launcher bootstrapping Warden & spawning 50 swarm agents
├── tests/                        # Component 7: Verification Test Suite
│   ├── __init__.py
│   ├── test_warden_hardware.py   # Tests Warden MIG tiers, survival tax, watchdog, unified memory
│   ├── test_data_forge.py        # Tests synthetic generation, KvikIO streaming, DALI loading
│   ├── test_physics.py           # Tests exact LOB slippage math, spread cost, portfolio accounting
│   ├── test_evolution_state.py   # Tests SelfMod rules, EWC Fisher matrix, Git rollback, PBT/HER
│   └── test_escrow_airgap.py     # Tests atomic escrow locking, 10x airgap splits, exact refund
├── docs/                         # Detailed Architectural Documentation
│   ├── ARCHITECTURE.md           # System design philosophy & high-level component interaction
│   ├── WARDEN_HYPERVISOR.md      # Deep dive: MIG/MVS tiers, taxes, unified memory, & OOM watchdog
│   ├── DATA_FORGE_AND_PHYSICS.md # Deep dive: KvikIO GPUDirect, DALI, & LOB friction physics
│   ├── SWARM_EVOLUTION.md        # Deep dive: SelfMod, EWC, Git rollback, & RL mechanics
│   ├── ESCROW_AND_AIRGAP.md      # Deep dive: Atomic P2P escrow & 10x flash-crash airgap
│   └── DEPLOYMENT_AND_TESTING.md # Production DGX setup & comprehensive test suite guide
├── data_store/                   # Partitioned historical LOB data (`symbol/year/month/day/`)
├── state/                        # Per-container SQLite ledgers, checkpoints, & ChromaDB
└── README.md                     # Project overview (this file)
```

---

## ⚡ Quick Start & Execution

### 1. Local Laptop Development & Simulation Mode
Project TradeJack is engineered with automatic binary and simulation fallbacks. When run on a standard development laptop without an NVIDIA DGX Spark or CUDA GPU, the system seamlessly initializes clean **CPU / Numpy / Polars / SQLite** fallbacks:

```bash
# Run the Master Genesis Prime Orchestrator (Simulates 10 competing sovereign agents locally)
python -m scripts.genesis_prime --containers 10 --steps 20
```

**Expected Console Report**:
```json
{
  "execution_time_sec": 13.09,
  "hardware_verification": {
    "torch_available": false,
    "cuda_available": false,
    "gpu_name": "None",
    "fp8_support": false
  },
  "total_containers_spawned": 10,
  "swarm_initial_capital": 100.0,
  "swarm_final_capital": 100.02,
  "top_performer": {
    "child_id": 1,
    "steps_completed": 20,
    "final_equity": 10.00405,
    "peak_equity": 10.03273,
    "max_drawdown": 0.00418,
    "sharpe_ratio": 10.5614,
    "sortino_ratio": 10.0208,
    "active_tier": 2,
    "active_model": "Dilated-CNN-Seq2seq"
  }
}
```

### 2. Running the Verification Suite
Run the comprehensive verification test suite to validate all 16 core subsystems across the 5 test modules:

```bash
python -m unittest discover -s tests -v
```

**Output**:
```
test_synthetic_generation_and_ingestion ... ok
test_kvikio_data_forge_streaming ... ok
test_dali_loader_batch_iteration ... ok
test_escrow_initiation_and_locking ... ok
test_validation_airgap_and_refund_on_rejection ... ok
test_airgap_validation_engine_splits ... ok
test_self_mod_engine_tier_rules ... ok
test_ewc_optimizer_fisher_computation ... ok
test_git_financial_rollback ... ok
test_rl_mechanics_her_and_pbt ... ok
test_lob_env_reset_and_step ... ok
test_portfolio_accounting_engine ... ok
test_mig_memory_slicing_and_tiers ... ok
test_logarithmic_stagnation_tax ... ok
test_oom_watchdog_penalty ... ok
test_unified_memory_allocator ... ok

----------------------------------------------------------------------
Ran 16 tests in 0.656s

OK
```

### 3. Production Deployment on NVIDIA DGX Spark (Blackwell)
To saturate the target **NVIDIA DGX Spark** server (`128GB Unified Memory`, `CUDA 13.0`):

1. **Build the Production Swarm Image**:
   ```bash
   docker build -t tradejack:child-cuda13 -f scripts/Dockerfile.child .
   ```
2. **Launch Master Genesis Prime across 50 MIG Partitions**:
   ```bash
   python -m scripts.genesis_prime --containers 50 --steps 1000000 --dgx-mode
   ```
   *The Warden automatically detects Grace Blackwell hardware, activates GPUDirect Storage (`cufile`), initializes zero-copy DALI pipelines, and enforces MIG/MVS hardware boundaries and NVLink-C2C unified memory swapping.*

---

## 📚 Documentation
For detailed architectural breakdowns of each subsystem, consult the `docs/` directory:
- [System Architecture Overview](docs/ARCHITECTURE.md)
- [Warden Hypervisor & Memory Controller](docs/WARDEN_HYPERVISOR.md)
- [High-Throughput Data Forge & LOB Physics Engine](docs/DATA_FORGE_AND_PHYSICS.md)
- [Sovereign Swarm Automaton & Neuro-Evolution](docs/SWARM_EVOLUTION.md)
- [Trustless P2P Escrow & 10x Validation Airgap](docs/ESCROW_AND_AIRGAP.md)
- [Deployment & Verification Guide](docs/DEPLOYMENT_AND_TESTING.md)

---

## 🛡️ License & Disclaimer
Project TradeJack is autonomous, self-modifying financial engineering software developed for high-performance computing research. Use at your own discretion.
