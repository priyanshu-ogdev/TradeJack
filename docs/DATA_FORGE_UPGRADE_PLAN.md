# Data Forge Upgrade Plan (Phase 2)

Phase 1 (previous pass) fixed four correctness bugs that were silently
breaking the pipeline and added a raw forex tick ingest leg. This plan covers
what's still needed before the RL agent can actually train on complete,
real, multi-asset data — in priority order, with the reasoning for that
order stated so it isn't just a task list.

## Priority order and why

1. **Forex feature engineering** — highest priority, because Phase 1 added
   raw FX ticks but nothing turns them into RL-consumable features. Data
   nobody can read is the same as no data.
2. **Sentiment oracle real ingestion** — second, because it's currently a
   storage layer with zero producers. An RL state vector that includes a
   sentiment slot fed by nothing will either crash or train against silent
   zeros, which is worse than omitting the feature.
3. **Cross-asset time alignment** — third, because it only matters once
   there's more than one real asset-class feature stream to align (crypto +
   forex + macro + sentiment) — which is true only after 1 and 2 land.
4. **Test coverage for everything new** — last on the list but not
   optional: nothing above counts as done until it's asserted, not just
   demoed via a `__main__` block.

## 1. Forex feature engineering (`forex_feature_engineering.py`)

FX ticks are two-sided quotes (bid, ask, bid_volume, ask_volume), not
trade prints, so the existing `feature_engineering.py` (built for Binance
aggTrades semantics) doesn't directly apply. Adapted features:

- **Spread** (`ask - bid`) and **relative spread** (`spread / mid`) — the FX
  equivalent of a direct liquidity-cost signal; crypto's OFI/VPIN infer cost
  from trade flow because spread data isn't reliably available at scale,
  but FX gives it directly.
- **Mid-price log-returns**, volume-bucketed like the existing OFI pipeline,
  reusing the same NaN-poisoning-safe clipping (`np.clip(mid, 1e-10, None)`)
  that `feature_engineering.py` already established.
- **Quote imbalance** (`(bid_volume - ask_volume) / (bid_volume + ask_volume)`)
  as the direct FX analog of OFI — a real ratio computable from the two-sided
  quote stream, not a forced reuse of the trade-flow formula.
- **Quote intensity** (ticks per minute) as a liquidity/session proxy — this
  is what lets a model learn the Tokyo/London/NY session structure without
  being told session boundaries explicitly.
- Session-gap handling: weekend/holiday gaps are real market closures, not
  missing data — features must not compute a "return" across a multi-day
  gap. Bucket boundaries respect a max-gap cutoff instead of blindly
  connecting the last Friday tick to the next Sunday tick.

Output: `processed/{PAIR}/fx_physics/YYYY/MM/DD/`, validated against a new
`ForexPhysicsSchema`.

## 2. Sentiment oracle real ingestion (`sentiment_source.py`)

The existing `sentiment_oracle.py` is a correctly-built Qdrant storage/query
layer — the fix is upstream of it, not inside it: a real, free producer.

- **Source**: free public RSS feeds (CoinDesk, CoinTelegraph, Investing.com
  markets, Reuters business) — no API key, no rate-limit auth, standard
  `feedparser` parsing. This is a deliberately modest first source: it will
  not match a paid Twitter/X firehose, and that limit is stated rather than
  glossed over.
- **Scoring**: VADER (`vaderSentiment`) — a free, local, lexicon-based
  sentiment scorer with no model download and no network call at score time.
  It is a weaker signal than a transformer embedding, but it's honest,
  reproducible, and doesn't silently depend on an unavailable service. The
  existing `SentimentOracle`'s embedding path (BGE-Large via Qdrant) stays
  as the "real embedding" upgrade path for later, once an embedding model is
  actually wired in — VADER is a stand-in that at least produces a real,
  non-empty signal today, not a placeholder that pretends to be the final
  design.
- **Output**: a compact per-symbol time series (`timestamp, headline_count,
  sentiment_mean, sentiment_std`) written to Parquet — cheap enough to join
  directly onto the physics features, unlike raw embeddings which need the
  vector DB.

## 3. Cross-asset time alignment (`time_alignment.py`)

- Resamples any set of per-asset feature frames onto one shared UTC grid at
  a configurable frequency.
- Explicit, stated gap policy (this was previously an unstated design gap):
  forward-fill up to a max staleness (config knob), beyond which a
  `stale`/`is_gap` flag column is set to 1 rather than silently
  forward-filling a multi-day FX weekend gap into fake "flat" price action.
  Crypto (which never truly gaps) will only ever hit the fill path for
  short outages; FX will legitimately hit the gap-flag path every weekend —
  that's real market structure the RL agent should be able to see, not
  hide.

## 4. Test coverage

Add `test_forex_features`, `test_sentiment_ingestion`, and
`test_time_alignment` to `tests/test_data_forge.py`, using small synthetic
fixtures (no network dependency) so they run in CI/offline the same way the
existing `test_synthetic_generation_and_ingestion` does.

## What's explicitly still out of scope after this plan

- A real embedding-based sentiment model (transformer + Qdrant vectors) —
  VADER is the honest interim signal, not the final one.
- A paid/consolidated crypto L2 depth source — still a real gap, see
  `docs/DATA_FORGE.md` §5.
- Live forex tick collection (Dukascopy is historical/free-tier tick
  download, not a live WebSocket) — if live FX forward-testing is needed
  later, that's a separate live-feed integration, not covered here.
