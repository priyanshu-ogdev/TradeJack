"""
Train and Promote Pipeline
==========================
Trains an SB3 reinforcement learning agent (e.g. PPO-DilatedCNN) on LOB data,
validates the candidate checkpoint against the Escrow Validation Airgap,
and promotes the weights to state/deployed/weights_promoted.zip for live paper trading.

Usage:
  python scripts/train_and_promote.py --timesteps 1024 --force
"""

import os
import sys
import json
import time
import shutil
import argparse
import logging
import numpy as np

# Ensure project root is on PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from swarm.rl_trainer import OnlineRLTrainer
from swarm.model_registry import REGISTRY
from physics.lob_env import TradeJackLOBEnv
from escrow.validation_airgap import ValidationAirgapEngine
from data_forge.kvikio_streamer import KvikIODataForge

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] (TrainPromote) %(message)s")
logger = logging.getLogger("TrainPromote")


def ensure_training_data(data_store: str, symbol: str = "BTC-USDT"):
    """Ensure structured partitioned parquet data exists for LOB env."""
    import asyncio
    from data_forge.parquet_ingest import ParquetIngestPipeline
    from data_forge.kvikio_streamer import KvikIODataForge

    forge = KvikIODataForge(data_store_dir=data_store, use_gds_if_available=False)
    partitions = forge.scan_available_partitions(symbol)
    if not partitions:
        logger.info(f"Generating synthetic partitioned LOB data in {data_store}...")
        ingest = ParquetIngestPipeline(data_store_dir=data_store)
        asyncio.run(ingest.generate_synthetic_crucible_data(
            symbol=symbol, num_days=2, ticks_per_day=600
        ))
        logger.info(f"Synthetic partitioned data ready in {data_store}")


def train_candidate(
    model_name: str,
    timesteps: int,
    symbol: str,
    state_dir: str,
    data_store: str,
) -> str:
    """Trains an RL model on LOB data and saves candidate weights."""
    ensure_training_data(data_store, symbol)

    # Initialize environment
    env = TradeJackLOBEnv(
        child_id=999,
        symbol=symbol,
        data_store_dir=data_store,
        seq_len=64,
        initial_cash=10.0,
    )

    log_dir = os.path.join(state_dir, "training_logs")
    os.makedirs(log_dir, exist_ok=True)

    logger.info(f"Initializing OnlineRLTrainer for '{model_name}' ({timesteps} timesteps)...")
    trainer = OnlineRLTrainer(
        model_name=model_name,
        env=env,
        device="cpu",
        log_dir=log_dir,
    )

    train_res = trainer.learn(total_timesteps=timesteps)
    logger.info(f"Training completed: {train_res}")

    # Save candidate weights
    cand_dir = os.path.join(state_dir, "candidates")
    os.makedirs(cand_dir, exist_ok=True)
    cand_path = os.path.join(cand_dir, f"{model_name}_candidate")
    trainer.save(cand_path)
    saved_path = cand_path if cand_path.endswith(".zip") else cand_path + ".zip"
    logger.info(f"Candidate model weights saved to {saved_path}")

    env.close()
    return saved_path


def promote_weights(
    weights_path: str,
    model_name: str,
    state_dir: str,
    airgap_result: dict,
):
    """Copies approved candidate weights to state/deployed/weights_promoted.zip."""
    deploy_dir = os.path.join(state_dir, "deployed")
    os.makedirs(deploy_dir, exist_ok=True)
    target_path = os.path.join(deploy_dir, "weights_promoted.zip")

    shutil.copy2(weights_path, target_path)
    logger.info(f"PROMOTED: {weights_path} -> {target_path}")

    # Log record
    log_file = os.path.join(deploy_dir, "promotion_log.jsonl")
    record = {
        "timestamp": time.time(),
        "model_name": model_name,
        "source_path": weights_path,
        "target_path": target_path,
        "airgap_passed": airgap_result.get("passed", False),
        "avg_sharpe": airgap_result.get("avg_sharpe", 0.0),
        "max_drawdown": airgap_result.get("max_drawdown", 0.0),
    }
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    logger.info(f"Logged promotion details to {log_file}")


def parse_args():
    p = argparse.ArgumentParser(description="TradeJack Train and Promote Pipeline")
    p.add_argument("--model", default="PPO-DilatedCNN", help="Model name in registry")
    p.add_argument("--timesteps", type=int, default=1024, help="Timesteps to train")
    p.add_argument("--symbol", default="BTC-USDT", help="Trading symbol")
    p.add_argument("--state-dir", default="state", help="State directory")
    p.add_argument("--data-store", default="data_store", help="Data store directory")
    p.add_argument("--force", action="store_true", help="Promote automatically")
    p.add_argument("--skip-airgap", action="store_true", help="Skip airgap evaluation")
    return p.parse_args()


def main():
    args = parse_args()
    logger.info("=" * 60)
    logger.info(f"Starting Train & Promote Pipeline for '{args.model}'")
    logger.info("=" * 60)

    # 1. Train candidate model
    cand_path = train_candidate(
        model_name=args.model,
        timesteps=args.timesteps,
        symbol=args.symbol,
        state_dir=args.state_dir,
        data_store=args.data_store,
    )

    # 2. Run Airgap Validation
    # NOTE: the previous default here was {"passed": True, "avg_sharpe": 1.25,
    # "max_drawdown": 0.05} — a FABRICATED passing result used whenever
    # --skip-airgap was set. Two real problems with that: (1) promote_weights()
    # writes this into promotion_log.jsonl, so the log would contain invented
    # performance numbers indistinguishable from a real evaluation after the
    # fact; (2) the promotion gate below is `if args.force or
    # airgap_result.get("passed", False)`, and a fabricated passed=True
    # satisfies that OR on its own — meaning --skip-airgap ALONE, without
    # --force, was already silently promoting. An honest "not evaluated" state
    # that cannot look like a pass is required instead.
    airgap_result = {"passed": False, "avg_sharpe": None, "max_drawdown": None, "note": "airgap_skipped_not_evaluated"}
    if not args.skip_airgap:
        logger.info(f"Evaluating candidate weights through ValidationAirgapEngine...")
        airgap = ValidationAirgapEngine(
            num_splits=5,
            min_required_sharpe=0.0,
            max_allowed_drawdown=0.25,
            data_store_dir=args.data_store,
        )
        airgap_result = airgap.evaluate_candidate_weights(cand_path, model_type=args.model)
        logger.info(f"Airgap results: {json.dumps(airgap_result, indent=2)}")
    else:
        logger.warning(
            "Airgap validation SKIPPED (--skip-airgap). airgap_result.passed is forced False — "
            "promotion will only proceed if --force is also passed explicitly."
        )

    # 3. Promotion Gate
    if airgap_result.get("passed", False) or (args.skip_airgap and args.force):
        promote_weights(cand_path, args.model, args.state_dir, airgap_result)
        logger.info("Train & Promote Pipeline finished successfully.")
    elif args.force and not args.skip_airgap:
        logger.warning("--force set but airgap was actually run and FAILED — promoting anyway per explicit --force.")
        promote_weights(cand_path, args.model, args.state_dir, airgap_result)
    else:
        logger.warning("Airgap criteria not met. Skipping promotion unless --force is specified.")


if __name__ == "__main__":
    main()
