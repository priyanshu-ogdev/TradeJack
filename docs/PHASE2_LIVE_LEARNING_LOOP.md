# Phase 2: Closing the Live-Learning Loop

Original problem statement (from the upgrade plan): "continuous" training meant
continuous re-training on the same synthetic-plus-stale-bulk data mix, cycle after
cycle, regardless of what actually happened in the live market during testnet/live
operation. Nothing that happened live ever fed back into retraining.

## What was actually missing, traced precisely

`execution/binance_live_feed.py`'s `BinanceLiveDepthFeed` (used by live trading)
already subscribes to Binance's **combined** `@depth@100ms/@aggTrade` stream — real
trade prints, with a genuine `is_buyer_maker` aggressor tag, already flow through
live trading sessions (`execution/streaming_features.py`'s `on_trade()` consumes
exactly this). But `data_forge/lob_collector.py` — the daemon meant to *persist*
data for later training use — only ever subscribed to `@depth@100ms`. Real trade
data was being observed live and thrown away, never written to disk.

Separately, `execution/streaming_features.py`'s own module docstring already named
the fix this needed: *"port TradeFlowPhysics's bucket logic to a streaming
form... as its own reviewed change to data_forge."* Rather than reimplement an
approximation of the batch bucketing math (remainder-carryover across days, the
Kyle's Lambda variance-floor fix, timestamp-unit auto-detection — all already
hardened in `TradeFlowPhysics`), this closes the gap by getting real captured data
into the exact format and location that pipeline already reads.

## What was built

1. **`data_forge/lob_collector.py`** now subscribes to the same combined stream as
   live trading and persists full-fidelity aggTrade records
   (`data_store/live/{symbol}/{date}/trades_HH.parquet`, sibling to the existing
   `depth_HH.parquet`) in exactly the column schema
   `TradeFlowPhysics._load_agg_trades()` expects from a Binance-Vision bulk
   download.

2. **`data_forge/live_trade_bridge.py`** (new): concatenates a day's captured trade
   files, de-duplicates on Binance's own `agg_trade_id` (not timestamp — multiple
   trades can share one), and writes the result to
   `data_store/raw/{symbol}/aggTrades/{date}/` — the exact directory
   `TradeFlowPhysics.process_daily_file()` already globs.

   **Correction from the first version of this doc**: I originally described the
   both-files-exist case as "`files[0]` picks one, not a merge" — true, but
   understated. `process_daily_file()`'s file selection had **no explicit sort at
   all** (`glob.glob()`'s order is OS/filesystem-dependent, not alphabetical), so
   which file got processed when both a bulk download and a live-bridged file
   existed for the same date was genuinely non-deterministic across runs, not
   just "some fixed but arbitrary choice." Found this while reviewing Phase 2
   rather than assuming the caveat I'd already written was precise enough — fixed
   directly in `feature_engineering.py`: file selection is now sorted
   deterministically, with a genuine bulk-download file always preferred over a
   live-bridged one when both exist (a completed historical archive is the more
   authoritative source when available). Verified with real string fixtures in
   both input orders, confirming the same bulk file wins regardless of glob's
   original ordering.

3. **`training/continuous_trainer.py`**: `_bridge_live_data()` runs the bridge +
   `TradeFlowPhysics.process_daily_file()` for both "today" and "yesterday" (UTC)
   at the top of every cycle (configurable via `bridge_live_data_every_cycles`).
   Wrapped in a broad `try/except` on purpose — verified directly, not assumed:
   in this sandbox (no `polars`/`pydantic_settings`), calling this method logs the
   failure and returns normally instead of crashing the training loop. A bad
   capture file or a missing dependency should cost this cycle's data freshness,
   never the whole continuous-training process.

4. **Data-freshness check on the promotion gate**
   (`escrow/validation_airgap.py`): `validate_promotion_candidate()` now checks
   `check_data_freshness()` first, before running the (expensive) airgap
   simulation. If the newest available processed data is older than
   `DeployConfig.max_training_data_staleness_days` (default 3, matching the
   existing `monitoring_out_of_band_days` default), promotion is refused with
   `reason="data_stale"` and the simulation doesn't even run. Rationale, stated
   directly: a promotion decision made against data that's stopped updating is
   the same failure mode as the force-promoted, statistically-rejected checkpoint
   this project's own history already found sitting in `state/deployed/` once —
   just quieter, because Sharpe/drawdown can look fine while measuring a market
   that no longer exists.

   Also fixed a small latent inconsistency caught while wiring this: the promotion
   gate accepted a `symbol` for the freshness check but the actual simulation call
   underneath it didn't forward that same symbol — silently defaulting to
   `"BTC-USDT"` regardless. Both now use the same value.

## What this does NOT claim to do

- **Does not guarantee the freshest processed file is picked up mid-cycle** by
  whatever agent is currently training. `_bridge_live_data()` runs before
  `agent.train()` each cycle, which is the most this layer can promise without
  inspecting `physics/lob_env.py`'s own partition-caching behavior — whether an
  already-constructed environment re-globs the processed directory per-episode or
  only at construction is that module's concern, not this bridge's.
- **Does not merge live-bridged and bulk-downloaded data for the same day** — see
  the stated caveat in `live_trade_bridge.py` above.
- **The freshness check measures "is there any recent data in the store at all,"
  not "did this specific evaluation window use it."** A coarser, more honest
  claim than "verified the exact split was fresh" — `TradeJackLOBEnv`'s partition
  walk doesn't currently expose which date a given split actually drew from.

## Verification status

Everything requiring `torch`/`gymnasium`/`stable_baselines3`/`polars`/
`pydantic_settings` (the actual WebSocket capture loop, the RL environment, the
airgap simulation itself) could not be executed in this sandbox — same limitation
as the rest of this project's `data_forge`/`physics`/`training` work. What *was*
actually run and verified here:

- The full record-mapping from a real Binance combined-stream aggTrade envelope
  shape to `TradeFlowPhysics`'s exact expected columns and types — traced and
  executed directly, not assumed.
- `_bridge_live_data()`'s crash-resilience: called it directly in an environment
  missing `polars`/`pydantic_settings` and confirmed it logs and returns
  normally rather than propagating — the actual failure mode it's meant to
  survive, reproduced for real, not just designed for on paper.
- `check_data_freshness()` / `latest_processed_data_date()`: four real filesystem
  scenarios (no data, fresh data, only-stale data, mixed old+new picking the
  newest) — all passed exactly as designed.
- Full existing test suite (48 tests across composition/predictor/tracker/composer)
  plus `risk_guardian.py`'s and `deploy_config.py`'s self-tests: still green,
  confirming none of this Phase 2 work regressed anything already verified.

Before trusting this against real capital: run `lob_collector.py` against real
Binance for at least a full day, confirm `trades_HH.parquet` files actually
accumulate with sane row counts, then run `live_trade_bridge.py` and
`TradeFlowPhysics.process_daily_file()` over that real day's output and manually
sanity-check the resulting `physics.parquet` against a known-good bulk-downloaded
day for the same symbol.
