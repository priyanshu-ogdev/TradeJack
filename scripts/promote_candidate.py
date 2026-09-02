# scripts/promote_candidate.py - Human-Confirmed Promotion CLI
# Gates Crucible->Deployment promotion through the airgap.
# Requires human confirmation before copying weights. No auto-promotion.
#
# Usage:
#   python scripts/promote_candidate.py --weights state/tournament/agent_0/model.zip
#                                       --model PPO-DilatedCNN
#                                       --child-id 0
#   Optionally: --force to skip confirmation prompt (for CI pipelines with human oversight)

import os
import sys
import json
import time
import shutil
import argparse
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] (Promoter) %(message)s")
logger = logging.getLogger("Promoter")

DEPLOY_WEIGHTS_DIR = "state/deployed"
PROMOTION_LOG = "state/deployed/promotion_log.jsonl"


def parse_args():
    parser = argparse.ArgumentParser(
        description="TradeJack Promotion CLI - gates Crucible->Deployment with human confirmation"
    )
    parser.add_argument("--weights", required=True, help="Path to candidate weights file (.pt)")
    parser.add_argument("--model", required=True, help="Model architecture name (e.g. SAC-DilatedCNN)")
    parser.add_argument("--child-id", type=int, required=True, help="Source child ID in the Crucible")
    parser.add_argument("--data-store", default="d:/TradeJack/data_store", help="Path to data store")
    parser.add_argument("--force", action="store_true", help="Skip interactive confirmation (still requires human intent)")
    return parser.parse_args()


def run_airgap_validation(weights_path: str, model_type: str, data_store: str) -> dict:
    logger.info(f"Running 10x validation airgap for {weights_path} ({model_type})...")
    try:
        sys.path.insert(0, os.path.abspath("."))
        from escrow.validation_airgap import ValidationAirgapEngine
        airgap = ValidationAirgapEngine(
            num_splits=10,
            min_required_sharpe=1.0,
            max_allowed_drawdown=0.15,
            data_store_dir=data_store
        )
        result = airgap.evaluate_candidate_weights(weights_path, model_type=model_type)
        return result
    except Exception as e:
        logger.error(f"Airgap validation failed: {e}")
        return {"passed": False, "error": str(e), "avg_sharpe": 0.0, "max_drawdown": 1.0}


def promote(weights_path: str, model_type: str, child_id: int, airgap_result: dict):
    os.makedirs(DEPLOY_WEIGHTS_DIR, exist_ok=True)
    # Determine file extension for the promoted checkpoint
    ext = os.path.splitext(weights_path)[1] or ".zip"
    dest = os.path.join(DEPLOY_WEIGHTS_DIR, f"weights_promoted{ext}")
    shutil.copy2(weights_path, dest)
    logger.info(f"Weights promoted: {weights_path} -> {dest}")

    record = {
        "promoted_at": time.time(),
        "source_child_id": child_id,
        "source_weights": weights_path,
        "model_type": model_type,
        "airgap_result": airgap_result,
        "promoted_by": "human_confirmed",
    }
    os.makedirs(os.path.dirname(PROMOTION_LOG), exist_ok=True)
    with open(PROMOTION_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    logger.info(f"Promotion logged to {PROMOTION_LOG}")


def main():
    args = parse_args()

    if not os.path.exists(args.weights):
        logger.error(f"Weights file not found: {args.weights}")
        sys.exit(1)

    # Run airgap
    result = run_airgap_validation(args.weights, args.model, args.data_store)

    print("\n" + "="*60)
    print("AIRGAP VALIDATION RESULT")
    print("="*60)
    print(json.dumps(result, indent=2))
    print("="*60)

    if not result.get("passed", False):
        logger.error("PROMOTION BLOCKED: Candidate failed validation airgap.")
        logger.error(f"  Avg Sharpe: {result.get('avg_sharpe', 0):.3f} (required >= 1.0)")
        logger.error(f"  Max Drawdown: {result.get('max_drawdown', 1):.1%} (required <= 15%)")
        sys.exit(2)

    logger.info("AIRGAP PASSED. Requesting human confirmation before promotion.")
    logger.info(f"  Child {args.child_id} weights: {args.weights}")
    logger.info(f"  Model: {args.model}")

    if not args.force:
        print("\n*** HUMAN CONFIRMATION REQUIRED ***")
        print("Type 'PROMOTE' to confirm deployment of these weights:")
        confirm = input("> ").strip()
        if confirm != "PROMOTE":
            logger.warning("Promotion cancelled by operator.")
            sys.exit(0)

    promote(args.weights, args.model, args.child_id, result)
    logger.info("PROMOTION COMPLETE. Update deploy_config.py frozen_model_path to point to weights_promoted.pt.")


if __name__ == "__main__":
    main()
