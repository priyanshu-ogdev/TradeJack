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
import signal
import atexit
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, List, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] (GenesisPrime) %(message)s")
logger = logging.getLogger("GenesisPrime")

# BUG FOUND BY ACTUALLY RUNNING --mode crucible, not by reading this file:
# every path below used to be a hardcoded Windows-style literal
# ("d:/TradeJack/..."). That's a leftover from the original dev machine and
# is not a valid absolute path on Linux/Mac (no drive-letter concept there) —
# os.path.join/os.path.abspath silently treat "d:/TradeJack/state" as a
# RELATIVE path segment and prepend the actual cwd, so on any non-Windows
# host (i.e. any real DGX/cloud deployment, which is this project's actual
# target) state silently landed in a path like
# "<cwd>/d:/TradeJack/state/tournament/..." instead of failing loudly.
# Confirmed by running `genesis_prime.py --mode crucible` and inspecting
# where it actually wrote checkpoints. PROJECT_ROOT below is the portable
# replacement — same layout (state/, data_store/), computed relative to this
# file instead of assuming a specific OS and drive letter.
#
# NOTE: the same "d:/TradeJack/..." literal appears as a default parameter
# value in ~20 other files across swarm/, warden/, escrow/, physics/, and
# tests/ (39 occurrences total, checked with grep). Those are lower priority
# than this file — genesis_prime.py is the one place that PASSES the literal
# explicitly (not just relies on an unused default), so it's the one that
# actually broke in a real run. The rest should get the same treatment in a
# follow-up pass rather than being fixed blind in this one.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

import os
import sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from warden.warden_core import WardenHypervisor
from warden.unified_memory_swap import BlackwellUnifiedAllocator
from warden.oom_watchdog import RecklessnessWatchdog
from warden.lineage_vector_db import LineageVectorDB
from warden.compute_server import WardenComputeServer
from warden.vllm_server import SharedBrainvLLM
from data_forge.parquet_ingest import ParquetIngestPipeline
from swarm.child_agent import SovereignChild


class GenesisPrimeLauncher:
    """
    Master deployment commander for Project TradeJack.
    """

    def __init__(self, workspace_dir: str = PROJECT_ROOT, num_containers: int = 50):
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
        self.compute_server: Optional[WardenComputeServer] = None
        self.vllm_server: Optional[SharedBrainvLLM] = None
        self.compute_thread: Optional[threading.Thread] = None

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
        Initializes the Warden Hypervisor, OOM Watchdog, Lineage Vector DB, Compute Server, and vLLM.
        """
        logger.info("================== BOOTING WARDEN HYPERVISOR ==================")
        # 1. Boot vLLM Shared Brain
        self.vllm_server = SharedBrainvLLM()
        self.vllm_server.launch_server()
        
        # 2. Boot Warden Compute API Server (runs Warden core + Watchdog + Audit Loop)
        self.compute_server = WardenComputeServer(port=8080, state_dir=self.state_dir, swarm_size=self.num_containers)
        self.compute_thread = threading.Thread(target=self.compute_server.start, daemon=True)
        self.compute_thread.start()
        
        # Extract references
        self.warden = self.compute_server.warden
        self.watchdog = self.compute_server.watchdog
        self.vector_db = LineageVectorDB(db_path=os.path.join(self.state_dir, "chroma_db"))
        
        time.sleep(2.0)
        logger.info(f"Warden infrastructure initialized and online for {self.num_containers} child endpoints.")

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

    async def run_genesis(self, max_steps: int = 1000, simulate_locally: bool = False):
        """
        Executes the master Crucible Loop.
        """
        logger.info("================== GENESIS PRIME IGNITION ==================")
        start_time = time.time()
        hw_report = self.verify_hardware_capabilities()
        
        # SOTA Fix Bug 5: Strict Environment Masking
        # Mask the main orchestrator process from the GPU to prevent VRAM fragmentation for vLLM
        if self.is_dgx_blackwell:
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            
        self.boot_warden_infrastructure()
        
        # SOTA Fix Bug 1: The Reaper Protocol
        def reap_zombies():
            logger.critical("REAPER PROTOCOL: Tearing down sub-services...")
            if self.vllm_server:
                self.vllm_server.stop_server()
            if self.compute_server and hasattr(self.compute_server, "_stop_event"):
                self.compute_server._stop_event.set()
                if hasattr(self.compute_server, '_audit_process') and self.compute_server._audit_process:
                    self.compute_server._audit_process.terminate()
                    self.compute_server._audit_process.join(timeout=2.0)
                    
        atexit.register(reap_zombies)
        signal.signal(signal.SIGINT, lambda s, f: reap_zombies() or sys.exit(0))
        signal.signal(signal.SIGTERM, lambda s, f: reap_zombies() or sys.exit(0))

        await self.bootstrap_data_forge()
        
        try:
            swarm_results = self.spawn_sovereign_swarm(simulate_locally=simulate_locally, max_steps_per_child=max_steps)
        finally:
            logger.info("================== SHUTTING DOWN WARDEN INFRASTRUCTURE ==================")
            reap_zombies()
            atexit.unregister(reap_zombies)
        
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
    import argparse

    parser = argparse.ArgumentParser(description="TradeJack v3 Genesis Prime Launcher")
    parser.add_argument("--mode", choices=["paper", "testnet", "live", "crucible"], default="crucible",
                        help="Execution mode: paper (simulated), testnet (Binance testnet), live (real money), crucible (training only)")
    parser.add_argument("--agents", type=int, default=4, help="Number of agents in tournament (3-10)")
    parser.add_argument("--max-steps", type=int, default=10000, help="Total timesteps per agent")
    parser.add_argument("--symbol", type=str, default="BTC-USDT", help="Trading pair")
    parser.add_argument("--capital", type=float, default=100.0, help="Starting capital (USDT)")
    parser.add_argument("--legacy", action="store_true", help="Use legacy 50-agent swarm mode (no SB3)")
    args = parser.parse_args()

    if args.legacy:
        # Legacy: Run old 50-container simulation
        launcher = GenesisPrimeLauncher(workspace_dir=PROJECT_ROOT, num_containers=args.agents)
        report = asyncio.run(launcher.run_genesis(simulate_locally=True, max_steps=args.max_steps))
        print("Genesis Prime Report:\n", json.dumps(report, indent=2))

    elif args.mode == "crucible":
        # v3: Run SB3 training tournament
        from training.crucible_tournament import CrucibleTournament
        logger.info(f"Starting Crucible Tournament: {args.agents} agents, {args.max_steps} timesteps")
        tournament = CrucibleTournament(
            data_store_dir=os.path.join(PROJECT_ROOT, "data_store"),
            state_dir=os.path.join(PROJECT_ROOT, "state", "tournament"),
            symbol=args.symbol,
        )
        results = tournament.run_tournament(
            total_timesteps=args.max_steps,
            cycle_timesteps=min(5000, args.max_steps // 4),
        )
        print("Tournament Results:\n", json.dumps(results, indent=2, default=str))

    elif args.mode in ("paper", "testnet", "live"):
        # v3: Run inference server with optional continuous training
        from scripts.deploy_config import DeploymentConfig
        from execution.live_inference_server import LiveInferenceServer
        from training.continuous_trainer import ContinuousTrainer

        config = DeploymentConfig(
            exchange_mode=args.mode,
            starting_capital=args.capital,
            model_name="PPO-DilatedCNN",
        )

        async def run_live():
            server = LiveInferenceServer.from_config(config)

            # Start continuous trainer in background
            # data_store_dir AND state_dir both anchored to PROJECT_ROOT --
            # leaving state_dir on its bare relative default while
            # data_store_dir was explicitly anchored was the other half of the
            # bug found on review (see ContinuousTrainer.__init__'s comment on
            # deployed_model_path): a process launched from any working
            # directory other than PROJECT_ROOT would have looked for its
            # promoted-model file and pending-promotion records somewhere
            # other than intended, even after deployed_model_path was fixed to
            # derive from state_dir -- deriving correctly from a wrong
            # default is still wrong.
            trainer = ContinuousTrainer(
                data_store_dir=os.path.join(PROJECT_ROOT, "data_store"),
                state_dir=os.path.join(PROJECT_ROOT, "state"),
                inference_server=server,
            )

            # Run both concurrently
            inference_task = asyncio.create_task(server.run_forever())
            training_task = asyncio.create_task(trainer.run_continuous())

            # PROCESS SUPERVISOR PREREQUISITE FIX: this used to rely on
            # `except KeyboardInterrupt` alone for graceful shutdown. That only
            # fires on SIGINT (Ctrl+C) -- Python's default handling of SIGTERM
            # (what `systemd stop`, `docker stop`, and scripts/process_supervisor.py
            # all send for a graceful stop) is immediate termination, which does
            # NOT raise KeyboardInterrupt and does NOT run server.stop()/trainer.stop().
            # In live/testnet mode that means open positions, risk-guardian state,
            # and the exchange connection could all be torn down mid-operation
            # instead of closed cleanly. A process supervisor that restarts this
            # process is only safe to build once a graceful stop signal actually
            # produces a graceful stop -- so this is fixed here, not in the
            # supervisor script, since the supervisor can't fix what the child
            # ignores.
            stop_event = asyncio.Event()

            def _handle_stop_signal(sig_name: str):
                logger.info(f"Received {sig_name} -- shutting down gracefully (server.stop() + trainer.stop()).")
                stop_event.set()

            loop = asyncio.get_running_loop()
            for sig in (signal.SIGINT, signal.SIGTERM):
                try:
                    loop.add_signal_handler(sig, _handle_stop_signal, sig.name)
                except NotImplementedError:
                    # add_signal_handler isn't available on Windows's default event
                    # loop -- fall back to the synchronous handler, which still
                    # covers SIGINT (Ctrl+C) there; Windows has no real SIGTERM
                    # equivalent for console apps anyway.
                    signal.signal(sig, lambda s, f: stop_event.set())

            async def _wait_for_stop():
                await stop_event.wait()
                server.stop()
                trainer.stop()

            stop_task = asyncio.create_task(_wait_for_stop())

            done, pending = await asyncio.wait(
                {inference_task, training_task, stop_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
            for task in pending:
                task.cancel()
            # Let cancellation actually propagate through each task's own
            # cleanup (finally blocks, etc.) before the process exits, rather
            # than firing cancel() and immediately falling through.
            await asyncio.gather(*pending, return_exceptions=True)
            for task in done:
                if task is not stop_task and task.exception() is not None:
                    raise task.exception()

        asyncio.run(run_live())
