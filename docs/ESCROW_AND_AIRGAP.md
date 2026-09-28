# Trustless P2P Escrow & Out-of-Sample Validation Airgap

The `escrow/` package prevents poisoned weights, overfitted architectures, and malicious parameter injections from spreading across the 50-container swarm during peer-to-peer weight exchange.

---

## 1. Trustless P2P Escrow Bridge (`escrow_contract.py`)

When an underperforming container (the buyer) discovers that a peer container (the seller) has achieved a superior Sharpe ratio via the social relay (`social_relay.py`), it can purchase a copy of the seller's neural weights using accumulated USDC cash reserves (`P2PEscrowBridge`).

Because containers operate as sovereign entities without mutual trust, transactions are governed by an atomic state machine:
```
[INITIATE_ESCROW] ---> (Lock USDC in Buyer Ledger) ---> [VALIDATION AIRGAP]
                                                                |
                          +-------------------------------------+-------------------------------------+
                          | (Passes Out-of-Sample Splits)                                         | (Fails Validation)
                          v                                                                           v
               [SETTLE_OR_REFUND: SETTLED]                                                [SETTLE_OR_REFUND: REFUNDED]
         (Transfer USDC to Seller & Weights to Buyer)                               (Refund 100% USDC to Buyer & Destroy Weights)
```

1. **Escrow Initiation (`initiate_escrow`)**: Checks the buyer's balance (`get_balance`). Deducts the purchase price (`price_usdc`) and places it into an atomic transaction lock (`status = "LOCKED"`). The seller provides a reference to the candidate weight file (`weights_path`).
2. **Settlement or Refund (`settle_or_refund`)**: Calls `ValidationAirgapEngine.evaluate_candidate_weights()`. If the candidate passes all out-of-sample tests, the bridge transfers the locked USDC to the seller (`SETTLED`) and grants weight ownership. If the candidate fails, the bridge issues an immediate **100% refund (`REFUNDED`)** back to the buyer's ledger.
3. **Asynchronous Non-Blocking Resolution**: Integrated with the Phase 5 `PortfolioAccountingEngine`, the escrow transactions are committed via Write-Ahead Logging (`PRAGMA journal_mode=WAL;`), ensuring that 50 active children can continuously initiate escrow trades without blocking the main trading engine or freezing the FastAPI Warden endpoints.

---

## 2. Out-of-Sample Validation Airgap (`validation_airgap.py`)

To ensure candidate weights perform well across varied market conditions rather than simply overfitting to one recent window, `ValidationAirgapEngine` sandboxes candidate models inside a configurable multi-split stress test (`num_splits`, default 10):

- **Seeded Random-Offset Splits**: Evaluates candidate weights (standard `stable-baselines3` `.zip` checkpoints) across `num_splits` windows into the data available under `data_store_dir`, each starting from a different seeded random tick offset. **This description reflects a fix, not the original design**: an earlier version seeded `env.reset(seed=seed)` per split but the underlying environment never actually consumed that seed to vary which data got played — `TradeJackLOBEnv._partition_streamer()` always walked `sorted(scan_available_partitions(...))` from the same starting point regardless of seed. Every "split" replayed the identical window, confirmed empirically (all 5-10 splits returned bit-for-bit identical Sharpe ratios to full float precision when this was tested). The fix adds a seeded random warm-up skip before each split's measured window, verified afterward to produce genuinely different results per split. There is no curated set of named historical crisis windows (COVID crash, a specific liquidation cascade, etc.) implemented anywhere in this file — building one would be a real, worthwhile upgrade on top of the current random-offset approach.
- **Strict Performance Cutoffs**: To clear the airgap (`passed = True`), candidate checkpoints must achieve across all splits:
  - **Average Sharpe Ratio >= `1.0`** (`min_required_sharpe`)
  - **Maximum Drawdown <= `15.0%`** (`max_allowed_drawdown`)
- **Poisoning Prevention**: Any model architecture or SB3 zip tensor collection that fails either condition is immediately flagged as overfitted or poisoned (`AIRGAP REJECTED`), preventing it from infecting other containers in the swarm.
- **Aggressive Sandbox Purging**: Once the validation evaluation completes, the sandbox immediately invokes the Reaper Protocol `gc.collect()` and `torch.cuda.empty_cache()` hooks to prevent the candidate weights from leaking into the VRAM pool.

### A second fixed bug worth knowing about: `scripts/train_and_promote.py --skip-airgap`

An earlier version of the manual training script defaulted to a **fabricated** airgap result (`{"passed": True, "avg_sharpe": 1.25, "max_drawdown": 0.05}`) whenever `--skip-airgap` was passed, and the promotion gate's `if args.force or airgap_result.get("passed", False)` check meant that fabricated `passed=True` alone — without `--force` — was already enough to trigger promotion. This has been fixed: skipping the airgap now produces an honest `{"passed": False, "note": "airgap_skipped_not_evaluated"}` that requires explicit `--force` to promote past, and the promotion log can no longer contain invented performance numbers indistinguishable from a real evaluation.
