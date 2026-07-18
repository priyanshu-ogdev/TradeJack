"""
Genesis Prime Master Launcher (`genesis_prime.py`).
Orchestrates the complete Project TradeJack autonomous deployment across:
1. Hardware Verification: Checks for NVIDIA Grace Blackwell (CUDA 13, 128GB Unified Memory) or falls back to Laptop simulation mode.
2. Warden Infrastructure: Boots `WardenHypervisor`, `compute_server.py` daemon, `SharedBrainvLLM`, and `LineageVectorDB`.
3. Data Forge Preparation: Ingests and partitions historical order book snapshots.
4. Swarm Genesis: Spawns 50 sovereign Child containers (or local multi-agent simulation threads) initialized with $10.00 equity,
   igniting the neuro-evolutionary competition toward $10,000.00.
"""

import os
import sys
import time
import json
import asyncio
import logging
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, List, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] (GenesisPrime) %(message)s")
logger = logging.getLogger("GenesisPrime")

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from warden.warden_core import WardenHypervisor
from warden.unified_memory_swap import BlackwellUnifiedAllocator
from warden.oom_watchdog import RecklessnessWatchdog
from warden.lineage_vector_db import LineageVectorDB
from data_forge.parquet_ingest import ParquetIngestPipeline
from swarm.child_agent import SovereignChild


class GenesisPrimeLauncher:
    """
    Master deployment commander for Project TradeJack.
    """

    def __init__(self, workspace_dir: str = "d:/TradeJack", num_containers: int = 50):
        self.workspace_dir = os.path.abspath(workspace_dir)
        self.num_containers = num_containers
        self.state_dir = os.path.join(self.workspace_dir, "state")
        self.data_store_dir = os.path.join(self.workspace_dir, "data_store")
        os.makedirs(self.state_dir, exist_ok=True)
        os.makedirs(self.data_store_dir, exist_ok=True)
        
        self.is_dgx_blackwell = False
        self.warden: Optional[WardenHypervisor] = None
        self.watchdog: Optional[RecklessnessWatchdog] = None
        self.vector_db: Optional[LineageVectorDB] = None

    def verify_hardware_capabilities(self) -> Dict[str, Any]:
        """
        Verifies GPU presence and Blackwell specific architecture markers (`CUDA 13`, `torch.float8_e4m3fn`).
        """
        logger.info("================== HARDWARE VERIFICATION ==================")
        report = {"torch_available": TORCH_AVAILABLE, "cuda_available": False, "gpu_name": "None", "fp8_support": False}
        
        if TORCH_AVAILABLE and torch.cuda.is_available():
            report["cuda_available"] = True
            report["gpu_name"] = torch.cuda.get_device_name(0)
            report["fp8_support"] = hasattr(torch, "float8_e4m3fn")
            self.is_dgx_blackwell = ("Blackwell" in report["gpu_name"] or "Spark" in report["gpu_name"] or report["fp8_support"])
            logger.info(f"[NVIDIA DGX DETECTED] GPU: {report['gpu_name']} | FP8 Native: {report['fp8_support']}")
        else:
            logger.info("[LAPTOP DEVELOPMENT MODE] No CUDA VRAM detected. Activating pure CPU/Numpy simulation pipeline.")
            self.is_dgx_blackwell = False
            
        return report

    def boot_warden_infrastructure(self):
        """
        Initializes the Warden Hypervisor, OOM Watchdog, and Lineage Vector DB.
        """
        logger.info("================== BOOTING WARDEN HYPERVISOR ==================")
        self.warden = WardenHypervisor(swarm_size=self.num_containers, enable_hardware_mig=self.is_dgx_blackwell)
        self.watchdog = RecklessnessWatchdog(state_dir=self.state_dir)
        self.vector_db = LineageVectorDB(db_path=os.path.join(self.state_dir, "chroma_db"))
        logger.info(f"Warden initialized with {self.num_containers} registered Child endpoints ($10.00 base equity each).")

    async def bootstrap_data_forge(self):
        """
        Generates or validates historical order book partition files in `data_store/`.
        """
        logger.info("================== BOOTSTRAPPING DATA FORGE ==================")
        ingest = ParquetIngestPipeline(data_store_dir=self.data_store_dir)
        # Check if BTC-USDT partitions exist
        btc_dir = os.path.join(self.data_store_dir, "BTC-USDT")
        if not os.path.exists(btc_dir) or not os.listdir(btc_dir):
            logger.info("No existing LOB partitions found. Generating high-precision synthetic Crucible data...")
            await ingest.generate_synthetic_crucible_data(symbol="BTC-USDT", num_days=3, ticks_per_day=150)
        else:
            logger.info(f"Verified existing LOB partition structures inside {btc_dir}.")

    def spawn_sovereign_swarm(self, simulate_locally: bool = True, max_steps_per_child: int = 100) -> List[Dict[str, Any]]:
        """
        Spawns the 50 sovereign agents.
        If on DGX Spark with Docker (`simulate_locally=False`), launches `docker run --gpus ...`.
        If on Laptop (`simulate_locally=True`), runs multi-threaded simulation across the 50 agents to test evolutionary dynamics.
        """
        logger.info(f"================== SPAWNING {self.num_containers} SOVEREIGN AGENTS ==================")
        results: List[Dict[str, Any]] = []
        
        if not simulate_locally and self.is_dgx_blackwell:
            logger.info("Executing production Docker container spawns across MIG partitions...")
            for idx in range(self.num_containers):
                container_name = f"tradejack_child_{idx}"
                cmd = [
                    "docker", "run", "-d", "--name", container_name,
                    "--net=host",
                    "-v", f"{self.workspace_dir}/data_store:/workspace/data_store",
                    "-v", f"{self.workspace_dir}/state/child_{idx}:/workspace/state/child_{idx}",
                    "tradejack:child_latest", "--child-id", str(idx)
                ]
                try:
                    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
                    logger.info(f"Launched container {container_name}")
                except Exception as e:
                    logger.error(f"Failed to spawn Docker container {container_name}: {e}")
            return results
        else:
            logger.info("Running multi-threaded Local Swarm Crucible Simulation across all 50 agents...")
            
            def run_single_child(child_id: int) -> Dict[str, Any]:
                try:
                    child = SovereignChild(child_id=child_id, symbol="BTC-USDT", initial_cash=10.0, state_dir=self.state_dir, data_store_dir=self.data_store_dir)
                    # Run loop
                    return child.run_crucible_loop(max_steps=max_steps_per_child)
                except Exception as e:
                    logger.error(f"Child {child_id} encountered exception: {e}")
                    return {"child_id": child_id, "error": str(e), "final_equity": 0.0}
                    
            with ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 4)) as executor:
                futures = [executor.submit(run_single_child, i) for i in range(self.num_containers)]
                for future in futures:
                    results.append(future.result())
                    
            logger.info("Local Swarm Crucible Simulation concluded across all 50 containers.")
            return results

    async def run_genesis(self, simulate_locally: bool = True, max_steps: int = 50) -> Dict[str, Any]:
        """
        Master execution sequence.
        """
        start_time = time.time()
        hw_report = self.verify_hardware_capabilities()
        self.boot_warden_infrastructure()
        await self.bootstrap_data_forge()
        
        swarm_results = self.spawn_sovereign_swarm(simulate_locally=simulate_locally, max_steps_per_child=max_steps)
        
        # Calculate aggregate swarm statistics
        total_eq = sum(r.get("final_equity", 10.0) for r in swarm_results)
        max_eq = max((r.get("final_equity", 10.0) for r in swarm_results), default=10.0)
        best_child = next((r for r in swarm_results if r.get("final_equity", 10.0) == max_eq), {})
        
        report = {
            "execution_time_sec": round(time.time() - start_time, 2),
            "hardware_verification": hw_report,
            "total_containers_spawned": len(swarm_results),
            "swarm_initial_capital": len(swarm_results) * 10.0,
            "swarm_final_capital": round(total_eq, 2),
            "top_performer": best_child,
            "sample_results": swarm_results[:3]
        }
        logger.info(f"================== GENESIS PRIME COMPLETE (Duration: {report['execution_time_sec']}s) ==================")
        logger.info(f"Swarm Capital Progression: ${report['swarm_initial_capital']:.2f} -> ${report['swarm_final_capital']:.2f} | Top Peak: ${max_eq:.2f} (Child {best_child.get('child_id')})")
        return report


if __name__ == "__main__":
    launcher = GenesisPrimeLauncher(workspace_dir="d:/TradeJack", num_containers=10)
    # Run a 10-container simulation for quick standalone verification
    report = asyncio.run(launcher.run_genesis(simulate_locally=True, max_steps=20))
    print("Genesis Prime Report:\n", json.dumps(report, indent=2))
