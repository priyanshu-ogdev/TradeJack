# Trustless P2P Escrow & 10x Validation Airgap

The `escrow/` package prevents poisoned weights, overfitted architectures, and malicious parameter injections from spreading across the 50-container swarm during peer-to-peer weight exchange.

---

## 1. Trustless P2P Escrow Bridge (`escrow_contract.py`)

When an underperforming container (the buyer) discovers that a peer container (the seller) has achieved a superior Sharpe ratio via the social relay (`social_relay.py`), it can purchase a copy of the seller's neural weights using accumulated USDC cash reserves (`P2PEscrowBridge`).

Because containers operate as sovereign entities without mutual trust, transactions are governed by an atomic state machine:
```
[INITIATE_ESCROW] ---> (Lock USDC in Buyer Ledger) ---> [VALIDATION AIRGAP]
                                                                |
                          +-------------------------------------+-------------------------------------+
                          | (Passes 10x Out-of-Sample Splits)                                         | (Fails Validation)
                          v                                                                           v
               [SETTLE_OR_REFUND: SETTLED]                                                [SETTLE_OR_REFUND: REFUNDED]
         (Transfer USDC to Seller & Weights to Buyer)                               (Refund 100% USDC to Buyer & Destroy Weights)
```

1. **Escrow Initiation (`initiate_escrow`)**: Checks the buyer's balance (`get_balance`). Deducts the purchase price (`price_usdc`) and places it into an atomic transaction lock (`status = "LOCKED"`). The seller provides a reference to the candidate weight file (`weights_path`).
2. **Settlement or Refund (`settle_or_refund`)**: Calls `ValidationAirgapEngine.evaluate_candidate_weights()`. If the candidate passes all out-of-sample tests, the bridge transfers the locked USDC to the seller (`SETTLED`) and grants weight ownership. If the candidate fails, the bridge issues an immediate **100% refund (`REFUNDED`)** back to the buyer's ledger.

---

## 2. 10x Out-of-Sample Validation Airgap (`validation_airgap.py`)

To ensure candidate weights perform well across real market volatility rather than simply overfitting to recent historical candles, `ValidationAirgapEngine` sandboxes candidate models inside a strict 10-split historical crucible:

- **Historical Flash-Crash Splits**: Evaluates candidate weights (`weights.pt` / `weights.npz`) across 10 severe historical regime shifts and liquidity crises (`2020-03-12` COVID crash, `2021-05-19` crypto liquidation cascade, etc.).
- **Strict Performance Cutoffs**: To clear the airgap (`passed = True`), candidate weights must achieve across all splits:
  - **Average Sharpe Ratio >= `1.0`** (`min_required_sharpe`)
  - **Maximum Drawdown <= `15.0%`** (`max_allowed_drawdown`)
- **Poisoning Prevention**: Any model architecture or weight tensor that fails either condition is immediately flagged as overfitted or poisoned (`AIRGAP REJECTED`), preventing it from infecting other containers in the swarm.
