# Project TradeJack: Anti-Gravity Systems Architecture

*Version: 1.0.0 (Genesis Prime Final)*

Project TradeJack is a fully orchestrated, autonomous, multi-agent reinforcement learning (MARL) trading hypervisor. Designed natively for a DGX Spark architecture, the system coordinates 50 sovereign child agents, pitting them in a survival-of-the-fittest crucible against a bare-metal execution environment. 

This document summarizes the final architecture, including all SOTA upgrades integrated during the 5 Phases of development.

---

## The 4 Pillars of TradeJack

### 1. Data Forge (The Zero-Copy Pipeline)
The Data Forge acts as the system's retina, ingesting terabytes of raw Level-3 Limit Order Book (LOB) data and transforming it into engineered features.
*   **KvikIO & GPUDirect Storage:** Bypasses the CPU entirely, streaming Parquet data directly from NVMe SSDs into GPU VRAM.
*   **Polars & CuDF:** Handles memory-mapped, zero-copy feature engineering, extracting micro-structure variables (e.g., Book Imbalance, VPIN, Order Flow Toxicity) at ultra-low latency.

### 2. Physics Engine (The Crucible)
The physics layer ensures agents learn in a mathematically pure representation of real-world friction.
*   **Asymmetric Slippage:** Simulates the exact market impact of aggressive vs. passive order executions based on deep LOB liquidity.
*   **Latency Delay Simulator:** Imposes random nanosecond network latency, forcing agents to predict future states rather than reacting to stale snapshots.
*   **Portfolio Tracker (SQLite WAL):** A fully asynchronous, non-blocking ledger engine that logs every trade, tax, and equity movement using Write-Ahead Logging (WAL) to prevent I/O starvation.

### 3. Sovereign Swarm (The Agents)
The 50 Child Agents operate as entirely sovereign processes/threads, competing for limited DGX VRAM and survival.
*   **Self-Modifying Architecture:** Powered by the `SelfModEngine`, agents dynamically swap their own PyTorch neural network architectures (e.g., from `Dilated-CNN-Seq2seq` to `Deep-Q-learning`) if the Warden degrades their VRAM capabilities.
*   **Aggressive CUDA Purge:** To prevent "Memory Creep" and VRAM leaks over 100-day epochs, the `SelfModEngine` explicitly unloads active C++ extensions, clears the Python `sys.modules` cache, and executes strict `torch.cuda.empty_cache()` sweeps during hot-swaps.
*   **Lineage Vector DB (Chroma):** Agents utilize a dedicated ChromaDB to log successful genome traits and adversarial strategies, performing RAG across generations.

### 4. Warden Hypervisor (The Enforcer)
The Warden is the system's merciless governor, designed to prevent complacency and enforce hardware constraints natively.
*   **Logarithmic Survival Tax:** An exponentially scaling tax that drains equity hourly based on how long the agent has been alive. Agents must generate continuous alpha or be purged via `INSOLVENCY_TAX_EXHAUSTION`.
*   **Stagnation Penalties:** Agents cannot sit on cash. The Warden tracks the `last_hwm_market_timestamp` (High Water Mark) and accelerates the tax if the agent fails to reach new equity peaks.
*   **Idempotent Multiprocessing Audit:** The Warden's Audit Loop (`run_audit_cycle`) executes in a dedicated OS-level `multiprocessing.Process` to bypass the Python GIL, utilizing `INSERT OR IGNORE` SQLite logic to guarantee taxes are never double-dipped during asynchronous race conditions.
*   **Hardware MIG Tiering:** Grants VRAM purely based on merit. Tier 1 (20GB) for high-equity agents, Tier 2 (4GB) for standard, and Tier 3 (1GB) for failing agents. 

---

## The Master Orchestrator: Genesis Prime
`genesis_prime.py` is the conductor of the orchestra. It ensures all services spin up and shut down without OS-level fragmentation.

### SOTA Orchestration Features
*   **The Reaper Protocol:** A strict `atexit` and `signal` handler mechanism that intercepts aborts (`Ctrl+C`) and gracefully tears down all subprocesses, preventing "Zombie" Docker containers from holding the VRAM hostage.
*   **Strict Environment Masking:** The orchestrator masks itself from the GPU (`CUDA_VISIBLE_DEVICES=""`), explicitly reserving the DGX's 128GB Unified Memory contiguous block for the vLLM Shared Brain and the PyTorch models.
*   **PagedAttention vLLM Server:** A background API endpoint serving Llama/DeepSeek weights to the agents for AST code-rewriting and reasoning, isolated perfectly from the SQLite I/O.

---

## The Execution Flow (The Crucible Loop)
1. **Boot:** `genesis_prime.py` validates DGX hardware, launches the `WardenComputeServer` in a background process, and boots the `SharedBrainvLLM`.
2. **Ingest:** The Data Forge memory-maps the order books.
3. **Spawn:** 50 Sovereign Children are launched, each starting with $10.00 and an identical Base RL model.
4. **Trade & Tax:** Agents execute trades against the Physics Engine. Concurrently, the Warden audits their SQLite ledgers, draining their cash via the Logarithmic Tax.
5. **Evolve & Degrade:** If an agent's equity drops, the Warden throttles its VRAM (e.g., Tier 2 -> Tier 3). The agent petitions vLLM for a lighter architecture, executes an Aggressive CUDA Purge, and resumes trading with a new neural net.
6. **Purge:** Agents that hit $0.00 are purged. The cycle continues until the surviving elite scale to $10,000.00 or are destroyed.

**Status: Project Complete. Systems Nominal. Deployment Ready.**
