# Production Deployment Guide & Comprehensive Verification Suite

This guide details how to build, deploy, and verify Project TradeJack across both local development laptops and physical **NVIDIA DGX Spark (Grace Blackwell, CUDA 13.0, 128GB Unified Memory)** hardware.

---

## 1. Running the Verification Test Suite (`tests/`)

Before deploying across physical hardware or running long-duration evolutionary runs, run the comprehensive verification suite to validate all 16 core subsystems across the 5 test modules:

```bash
python -m unittest discover -s tests -v
```

### Verification Suite Matrix
| Module | Test Method | Description | Expected Outcome |
| :--- | :--- | :--- | :---: |
| **`test_warden_hardware.py`** | `test_mig_memory_slicing_and_tiers` | Verifies dynamic assignment to Tier 1 (20GB), Tier 2 (4GB), Tier 3 (0GB) based on live Sharpe/Drawdown metrics. | `ok` |
| **`test_warden_hardware.py`** | `test_logarithmic_stagnation_tax` | Verifies `apply_survival_tax()` formula and deduction from SQLite ledger. | `ok` |
| **`test_warden_hardware.py`** | `test_oom_watchdog_penalty` | Verifies `RecklessnessWatchdog` CUDA OOM detection, -$10 penalty, and 24h Tier 3 lock. | `ok` |
| **`test_warden_hardware.py`** | `test_unified_memory_allocator` | Verifies `BlackwellUnifiedAllocator` NVLink-C2C pinning and zero-copy swapping. | `ok` |
| **`test_data_forge.py`** | `test_synthetic_generation_and_ingestion` | Verifies `ParquetIngestPipeline` synthetic generation and directory structure. | `ok` |
| **`test_data_forge.py`** | `test_kvikio_data_forge_streaming` | Verifies `KvikIODataForge` GPUDirect/Polars/NPZ tensor loading. | `ok` |
| **`test_data_forge.py`** | `test_dali_loader_batch_iteration` | Verifies `create_lob_dataloader` DALI batch iterator and feature dimensions. | `ok` |
| **`test_physics.py`** | `test_lob_env_reset_and_step` | Verifies `TradeJackLOBEnv` slippage math, spread cost across 8 LOB depth tiers. | `ok` |
| **`test_physics.py`** | `test_portfolio_accounting_engine` | Verifies `PortfolioAccountingEngine` real-time Sharpe/Sortino/Drawdown recording. | `ok` |
| **`test_evolution_state.py`** | `test_self_mod_engine_tier_rules` | Verifies `SelfModEngine` memory-bound architecture swapping (`Attention` vs `CNN` vs `Q-Learning`). | `ok` |
| **`test_evolution_state.py`** | `test_ewc_optimizer_fisher_computation` | Verifies `ElasticWeightConsolidation` diagonal Fisher Information computation. | `ok` |
| **`test_evolution_state.py`** | `test_git_financial_rollback` | Verifies `GitFinancialRollback` high-water mark tags and instant rollback on >15% drawdown. | `ok` |
| **`test_evolution_state.py`** | `test_rl_mechanics_her_and_pbt` | Verifies Hindsight Experience Replay sampling and Population-Based Training exploitation. | `ok` |
| **`test_escrow_airgap.py`** | `test_escrow_initiation_and_locking` | Verifies `P2PEscrowBridge` atomic USDC cash balance locking. | `ok` |
| **`test_escrow_airgap.py`** | `test_validation_airgap_and_refund_on_rejection` | Verifies 10x Airgap stress testing and 100% refund when weights fail validation. | `ok` |
| **`test_escrow_airgap.py`** | `test_airgap_validation_engine_splits` | Verifies out-of-sample evaluation across historical flash-crash splits. | `ok` |

---

## 2. Local Laptop Development & Simulation Mode

When executed on a development machine without CUDA hardware, the `GenesisPrimeLauncher` detects the environment and activates its CPU/Numpy/Polars binary fallback pipeline:

```bash
python -m scripts.genesis_prime --containers 10 --steps 50
```

---

## 3. Production Deployment on NVIDIA DGX Spark (Grace Blackwell)

### Step 1: Build Production Docker Image (`Dockerfile.child`)
The production container is pre-configured with CUDA 13, PyTorch 2.5+, cuDF, KvikIO, and DALI:

```bash
docker build -t tradejack:child-cuda13 -f scripts/Dockerfile.child .
```

### Step 2: Launch Master Genesis Prime across 50 MIG Containers
Run the orchestrator on the DGX server:

```bash
python -m scripts.genesis_prime --containers 50 --steps 1000000 --dgx-mode
```

- The Warden automatically detects Grace Blackwell hardware (`is_dgx_blackwell = True`).
- Activates GPUDirect Storage (`cufile`) for high-speed NVME-to-VRAM tensor streaming.
- Enforces MIG / MVS hardware boundaries across the 50 Docker containers and manages zero-copy NVLink-C2C unified memory swapping.
