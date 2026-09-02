#!/usr/bin/env python
"""
run_all_checks.py — Single-command full system verification.

Validates every module in TradeJack v3 can import, connect, and function correctly.
Run this after any code change to catch broken connections early.

Usage:
    python scripts/run_all_checks.py              # Full check (all phases)
    python scripts/run_all_checks.py --fast        # Quick import + unit test only
    python scripts/run_all_checks.py --phase 1     # Only check Phase 1
"""

import os
import sys
import time
import traceback
import argparse
from typing import List, Tuple, Dict, Any

# Project root
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# ─── Color codes ─────────────────────────────────────────────────────────────
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
RESET = "\033[0m"
BOLD = "\033[1m"


def check(name: str, func, results: list):
    """Run a single check and record result."""
    try:
        func()
        results.append((name, True, ""))
        print(f"  {GREEN}[OK]{RESET} {name}")
    except Exception as e:
        results.append((name, False, str(e)))
        print(f"  {RED}[FAIL]{RESET} {name}: {e}")


def phase_header(phase: str):
    print(f"\n{BOLD}{CYAN}{'='*60}")
    print(f"  {phase}")
    print(f"{'='*60}{RESET}")


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 0: Dependency Verification
# ═══════════════════════════════════════════════════════════════════════════════

def check_phase_0(results):
    phase_header("PHASE 0: Dependencies & Configuration")

    def check_torch():
        import torch
        assert torch.__version__, "torch has no version"

    def check_sb3():
        from stable_baselines3 import PPO, SAC, DQN
        assert PPO is not None

    def check_gymnasium():
        import gymnasium
        assert gymnasium.__version__, "gymnasium has no version"

    def check_ccxt():
        import ccxt
        assert ccxt.__version__, "ccxt has no version"

    def check_numpy():
        import numpy as np
        assert np.__version__

    def check_pyproject():
        assert os.path.exists(os.path.join(ROOT, "pyproject.toml"))

    def check_env_example():
        assert os.path.exists(os.path.join(ROOT, ".env.example"))

    def check_gitignore():
        gitignore = os.path.join(ROOT, ".gitignore")
        assert os.path.exists(gitignore)
        with open(gitignore) as f:
            assert ".env" in f.read()

    check("PyTorch installed", check_torch, results)
    check("stable-baselines3 installed", check_sb3, results)
    check("gymnasium installed", check_gymnasium, results)
    check("ccxt installed", check_ccxt, results)
    check("numpy installed", check_numpy, results)
    check("pyproject.toml exists", check_pyproject, results)
    check(".env.example exists", check_env_example, results)
    check(".gitignore has .env", check_gitignore, results)


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 1: RL Training Engine
# ═══════════════════════════════════════════════════════════════════════════════

def check_phase_1(results):
    phase_header("PHASE 1: RL Training Engine")

    def check_shared_encoder():
        from swarm.shared_encoder import LOBFeatureEncoder, LOBFeatureExtractor
        import torch
        enc = LOBFeatureEncoder(5, 4, channels=64, dilations=(1, 2, 4, 8), features_dim=128)
        out = enc(torch.randn(2, 64, 5), torch.randn(2, 4))
        assert out.shape == (2, 128), f"Expected (2,128), got {out.shape}"

    def check_model_registry():
        from swarm.model_registry import REGISTRY
        assert len(REGISTRY.cards) >= 4, f"Only {len(REGISTRY.cards)} models"
        assert "PPO-DilatedCNN" in REGISTRY.cards
        assert "SAC-DilatedCNN" in REGISTRY.cards
        assert "DuelingDQN" in REGISTRY.cards

    def check_registry_build():
        import asyncio
        from data_forge.parquet_ingest import ParquetIngestPipeline
        from physics.lob_env import TradeJackLOBEnv
        from swarm.model_registry import REGISTRY

        store = os.path.join(ROOT, "data_store")
        if not os.path.exists(os.path.join(store, "BTC-USDT")):
            ingest = ParquetIngestPipeline(data_store_dir=store)
            asyncio.run(ingest.generate_synthetic_crucible_data("BTC-USDT", 1, 200))

        env = TradeJackLOBEnv("BTC-USDT", 10.0, store, child_id=998)
        model = REGISTRY.build_model("PPO-DilatedCNN", env=env, device="cpu")
        assert model is not None

    def check_rl_trainer():
        from swarm.rl_trainer import OnlineRLTrainer

    def check_replay_buffer():
        from swarm.replay_buffer import PrioritizedReplayBuffer
        import numpy as np
        buf = PrioritizedReplayBuffer(capacity=50)
        buf.push(np.zeros((64, 5)), np.array([0.5, 0, 0, 0]), 1.0, 1.0, np.zeros((64, 5)), np.array([0.5, 0, 0, 0]), False, 1.0)
        assert len(buf) == 1

    def check_baselines():
        from swarm.baselines import BuyAndHoldBaseline, MomentumBaseline, MeanReversionBaseline
        import numpy as np
        obs = {"lob_sequence": np.random.rand(64, 5).astype(np.float32),
               "portfolio_state": np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float32)}
        for cls in [BuyAndHoldBaseline, MomentumBaseline, MeanReversionBaseline]:
            a = cls().predict(obs)
            assert -1.0 <= a <= 1.0, f"{cls.__name__} out of range: {a}"

    def check_rl_mechanics():
        from swarm.rl_mechanics import HindsightExperienceReplay, PopulationBasedTrainingEngine, AdversarialGANSpoofer
        import numpy as np
        her = HindsightExperienceReplay(100)
        her.push(np.zeros(8), [0.5], 0.1, np.ones(8), 9.5, 11.0, False)
        assert len(her) == 1

    def check_lob_env_gymnasium():
        import asyncio, gymnasium
        from physics.lob_env import TradeJackLOBEnv
        from data_forge.parquet_ingest import ParquetIngestPipeline
        store = os.path.join(ROOT, "data_store")
        if not os.path.exists(os.path.join(store, "BTC-USDT")):
            ingest = ParquetIngestPipeline(data_store_dir=store)
            asyncio.run(ingest.generate_synthetic_crucible_data("BTC-USDT", 1, 200))
        env = TradeJackLOBEnv("BTC-USDT", 10.0, store, child_id=997)
        obs, info = env.reset()
        assert "lob_sequence" in obs
        assert "portfolio_state" in obs
        obs2, rew, term, trunc, info2 = env.step([0.5])
        assert "equity" in info2

    def check_self_mod_manager():
        from swarm.self_mod_manager import SelfModEngine

    def check_deploy_config():
        from scripts.deploy_config import DeploymentConfig, DEPLOY_CONFIG
        assert DEPLOY_CONFIG.exchange_mode in ("paper", "testnet", "live")
        assert DEPLOY_CONFIG.starting_capital > 0

    check("shared_encoder imports + forward", check_shared_encoder, results)
    check("model_registry v3 models", check_model_registry, results)
    check("REGISTRY.build_model (PPO-DilatedCNN)", check_registry_build, results)
    check("rl_trainer imports", check_rl_trainer, results)
    check("replay_buffer add/len", check_replay_buffer, results)
    check("baselines predict", check_baselines, results)
    check("rl_mechanics HER/PBT/Spoofer", check_rl_mechanics, results)
    check("LOB env Gymnasium compliance", check_lob_env_gymnasium, results)
    check("self_mod_manager imports", check_self_mod_manager, results)
    check("deploy_config valid", check_deploy_config, results)


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 2: Execution Pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def check_phase_2(results):
    phase_header("PHASE 2: Execution Pipeline")

    def check_exchange_adapter():
        from execution.exchange_adapter import ExchangeAdapter, BinanceSpotAdapter, OrderResult, OrderBook

    def check_paper_exchange():
        import asyncio
        from execution.paper_exchange import PaperExchangeAdapter
        async def _t():
            p = PaperExchangeAdapter(initial_balance_usdt=100.0)
            await p.connect()
            r = await p.place_market_order("BTC/USDT", "buy", 0.001)
            assert r.status == "filled"
            assert r.fee > 0
            await p.close()
        asyncio.run(_t())

    def check_risk_guardian():
        import asyncio
        from execution.paper_exchange import PaperExchangeAdapter
        from execution.risk_guardian import RiskGuardian
        async def _t():
            ex = PaperExchangeAdapter(100.0)
            await ex.connect()
            rg = RiskGuardian(exchange=ex, starting_equity=100.0)
            rg.update_equity(100.0)
            allowed, reason = rg.check_order_allowed("buy", 0.0001, 60000.0)
            assert allowed, f"Should be allowed: {reason}"
            summary = rg.get_risk_summary()
            assert "equity" in summary
            await ex.close()
        asyncio.run(_t())

    def check_position_throttle():
        import numpy as np
        from execution.position_throttle import PositionThrottle
        t = PositionThrottle()
        for _ in range(50):
            t.record_return(np.random.normal(0.001, 0.005))
        f = t.get_throttled_fraction()
        assert 0 < f <= 1.0

    def check_live_inference_server():
        from execution.live_inference_server import LiveInferenceServer

    check("exchange_adapter imports", check_exchange_adapter, results)
    check("paper_exchange buy/sell", check_paper_exchange, results)
    check("risk_guardian check_order", check_risk_guardian, results)
    check("position_throttle compute", check_position_throttle, results)
    check("live_inference_server imports", check_live_inference_server, results)


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 3: Self-RL Training Loop
# ═══════════════════════════════════════════════════════════════════════════════

def check_phase_3(results):
    phase_header("PHASE 3: Self-RL Training Loop")

    def check_walk_forward():
        from training.walk_forward_evaluator import WalkForwardEvaluator
        ev = WalkForwardEvaluator()
        assert ev.min_trade_count == 20
        assert ev.significance_level == 0.05

    def check_crucible_tournament():
        from training.crucible_tournament import CrucibleTournament, DEFAULT_ROSTER
        assert len(DEFAULT_ROSTER) >= 3

    def check_continuous_trainer():
        from training.continuous_trainer import ContinuousTrainer

    def check_genesis_cli():
        import subprocess
        result = subprocess.run(
            [sys.executable, os.path.join(ROOT, "scripts", "genesis_prime.py"), "--help"],
            capture_output=True, text=True, cwd=ROOT
        )
        assert result.returncode == 0
        assert "--mode" in result.stdout
        assert "paper" in result.stdout

    def check_validation_airgap():
        from escrow.validation_airgap import ValidationAirgapEngine
        ve = ValidationAirgapEngine(num_splits=3)
        assert ve.num_splits == 3

    check("walk_forward_evaluator imports", check_walk_forward, results)
    check("crucible_tournament imports", check_crucible_tournament, results)
    check("continuous_trainer imports", check_continuous_trainer, results)
    check("genesis_prime --help", check_genesis_cli, results)
    check("validation_airgap v3 imports", check_validation_airgap, results)


# ═══════════════════════════════════════════════════════════════════════════════
# Phase L: Legacy Module Compatibility
# ═══════════════════════════════════════════════════════════════════════════════

def check_legacy(results):
    phase_header("LEGACY: Backward Compatibility")

    def check_warden_core():
        from warden.warden_core import WardenHypervisor

    def check_data_forge():
        from data_forge.parquet_ingest import ParquetIngestPipeline
        from data_forge.kvikio_streamer import KvikIODataForge

    def check_escrow():
        from escrow.escrow_contract import P2PEscrowBridge
        from escrow.validation_airgap import ValidationAirgapEngine

    def check_portfolio_tracker():
        from physics.portfolio_tracker import PortfolioAccountingEngine

    def check_child_agent():
        from swarm.child_agent import SovereignChild

    check("warden_core imports", check_warden_core, results)
    check("data_forge imports", check_data_forge, results)
    check("escrow imports", check_escrow, results)
    check("portfolio_tracker imports", check_portfolio_tracker, results)
    check("child_agent imports", check_child_agent, results)


# ═══════════════════════════════════════════════════════════════════════════════
# Cross-Module Connection Verification
# ═══════════════════════════════════════════════════════════════════════════════

def check_connections(results):
    phase_header("CROSS-MODULE: Connection Verification")

    def check_registry_to_env():
        """Registry → Env → SB3 model full pipeline."""
        import asyncio
        from data_forge.parquet_ingest import ParquetIngestPipeline
        from physics.lob_env import TradeJackLOBEnv
        from swarm.model_registry import REGISTRY
        store = os.path.join(ROOT, "data_store")
        if not os.path.exists(os.path.join(store, "BTC-USDT")):
            ingest = ParquetIngestPipeline(data_store_dir=store)
            asyncio.run(ingest.generate_synthetic_crucible_data("BTC-USDT", 1, 200))
        env = TradeJackLOBEnv("BTC-USDT", 10.0, store, child_id=996)
        model = REGISTRY.build_model("PPO-DilatedCNN", env=env, device="cpu")
        model.learn(total_timesteps=64)
        obs, _ = env.reset()
        action, _ = model.predict(obs, deterministic=True)
        assert action is not None

    def check_config_to_inference():
        """DeploymentConfig → LiveInferenceServer.from_config()."""
        from scripts.deploy_config import DeploymentConfig
        from execution.live_inference_server import LiveInferenceServer
        cfg = DeploymentConfig(exchange_mode="paper", starting_capital=100.0)
        server = LiveInferenceServer.from_config(cfg)
        status = server.get_status()
        assert status["model"] == "PPO-DilatedCNN"
        assert status["running"] == False

    def check_guardian_to_exchange():
        """RiskGuardian → PaperExchange → OrderResult pipeline."""
        import asyncio
        from execution.paper_exchange import PaperExchangeAdapter
        from execution.risk_guardian import RiskGuardian
        async def _t():
            ex = PaperExchangeAdapter(100.0)
            await ex.connect()
            rg = RiskGuardian(exchange=ex, starting_equity=100.0, min_hold_ticks=0)
            rg.update_equity(100.0)
            result = await rg.execute_safe_order("buy", 0.0005, 60000.0)
            assert result is not None
            assert result.status == "filled"
            await ex.close()
        asyncio.run(_t())

    def check_pbt_to_registry():
        """PBT → REGISTRY.get_model_card() and list_models_for_tier()."""
        from swarm.rl_mechanics import PopulationBasedTrainingEngine
        from swarm.model_registry import REGISTRY
        pbt = PopulationBasedTrainingEngine(swarm_size=4, exploit_fraction=0.5)
        status = [
            {"child_id": 0, "equity": 20.0, "sortino_ratio": 3.0, "ticks_active": 100,
             "learning_rate": 3e-4, "model_name": "PPO-DilatedCNN", "tier": 2},
            {"child_id": 1, "equity": 5.0, "sortino_ratio": -1.0, "ticks_active": 100,
             "learning_rate": 3e-4, "model_name": "DuelingDQN", "tier": 3},
        ]
        result = pbt.execute_pbt_step(status)
        assert len(result) == 2

    check("Registry → Env → SB3 train+predict", check_registry_to_env, results)
    check("Config → InferenceServer.from_config", check_config_to_inference, results)
    check("Guardian → PaperExchange → OrderResult", check_guardian_to_exchange, results)
    check("PBT → Registry model cards", check_pbt_to_registry, results)


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="TradeJack v3 Full System Verification")
    parser.add_argument("--fast", action="store_true", help="Quick import-only checks")
    parser.add_argument("--phase", type=int, choices=[0, 1, 2, 3], help="Run only a specific phase")
    args = parser.parse_args()

    results: List[Tuple[str, bool, str]] = []
    start = time.time()

    print(f"\n{BOLD}{'='*60}")
    print(f"  TradeJack v3 — Full System Verification")
    print(f"{'='*60}{RESET}")

    if args.phase is not None:
        {0: check_phase_0, 1: check_phase_1, 2: check_phase_2, 3: check_phase_3}[args.phase](results)
    elif args.fast:
        check_phase_0(results)
        check_legacy(results)
    else:
        check_phase_0(results)
        check_phase_1(results)
        check_phase_2(results)
        check_phase_3(results)
        check_legacy(results)
        check_connections(results)

    # ─── Summary ──────────────────────────────────────────────────────────
    elapsed = time.time() - start
    passed = sum(1 for _, ok, _ in results if ok)
    failed = sum(1 for _, ok, _ in results if not ok)
    total = len(results)

    print(f"\n{BOLD}{'='*60}")
    print(f"  RESULTS: {passed}/{total} passed, {failed} failed  ({elapsed:.1f}s)")
    print(f"{'='*60}{RESET}")

    if failed > 0:
        print(f"\n{RED}Failed checks:{RESET}")
        for name, ok, err in results:
            if not ok:
                print(f"  {RED}[FAIL]{RESET} {name}")
                print(f"    {err}")
        sys.exit(1)
    else:
        print(f"\n{GREEN}{BOLD}All checks passed! [OK]{RESET}")
        sys.exit(0)


if __name__ == "__main__":
    main()
