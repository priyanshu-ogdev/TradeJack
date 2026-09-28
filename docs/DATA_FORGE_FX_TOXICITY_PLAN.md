# FX Toxicity / Price-Impact Features — Design Plan (Phase 3)

## The problem with reusing crypto's VPIN/Kyle's Lambda directly

`feature_engineering.py`'s VPIN and Kyle's Lambda both start from **signed trade
volume**: Binance aggTrades tags every trade with `is_buyer_maker`, so the
aggressor side is known exactly (`trade_direction = -1 if is_buyer_maker else 1`).
That signed volume is what both metrics are built from.

Dukascopy's forex ticks are **quotes, not trades** — bid, ask, bid_volume,
ask_volume, with no aggressor tag at all, because there is no single trade to
tag. `quote_imbalance` (added in Phase 2) is a real, useful signal, but it's
a *quoted-depth* imbalance, not a *traded-flow* imbalance — computing VPIN or
Kyle's Lambda directly from it would be answering "who's quoting more size"
when the actual questions (informed-trading toxicity, price impact per unit
of flow) are about "who's actually moving the market." Feeding the crypto
formulas quote_imbalance instead of real signed trade flow would produce a
number shaped like VPIN without meaning what VPIN means — the failure mode
this plan exists to avoid.

## The fix: Bulk Volume Classification (BVC)

Easley, López de Prado & O'Hara (2012) designed exactly this fix for exactly
this situation — inferring buy/sell volume when only price and volume are
observable, no trade-direction tag available. That's a direct match for
Dukascopy's tick shape, so this plan adopts BVC rather than inventing a
weaker ad hoc proxy:

```
z_t        = price_change_t / rolling_std(price_change, window=W)
buy_frac_t = Φ(z_t)                       # Φ = standard normal CDF
buy_volume_t  = volume_t * buy_frac_t
sell_volume_t = volume_t - buy_volume_t
```

`volume_t` here is `bid_volume + ask_volume` per tick (total quoted size as
the closest available proxy for tick-level activity — Dukascopy doesn't
publish executed size either, only quoted size, which is a real limitation
worth stating plainly rather than dressing up as equivalent to trade volume).

This is a probabilistic classification (large price moves get classified as
strongly buy- or sell-dominated; near-zero moves split close to 50/50) rather
than a binary tick rule, which is why it was the standard adopted for
quote-only/OTC markets in the original VPIN literature — a binary uptick/
downtick rule throws away the magnitude information BVC keeps.

## What gets computed, mirroring the crypto module's already-reviewed math

Reusing `feature_engineering.py`'s numerically-stable rolling-OLS pattern
(BUG-16's fix: `cov(price_change, ofi) / var(ofi)` with a volume-scaled
variance floor, not naive division) for consistency and because that
formula was already hardened against near-zero-variance blowups:

1. **Volume buckets** (not time buckets) — cumulative `bid_volume + ask_volume`
   chunked into fixed-size buckets, same volume-clock convention VPIN was
   originally defined on. This is a different bucketing axis from Phase 2's
   `forex_feature_engineering.py` (which buckets by wall-clock time for
   spread/quote_imbalance/session features) — toxicity metrics need volume
   time, not clock time, or they lose their defining property (bucket sizes
   that adapt to how busy the market is).
2. **`bvc_vpin`** — `rolling_mean(|buy_volume - sell_volume|, 50) / rolling_mean(volume, 50)`,
   identical shape to crypto's `vpin_50`.
3. **`kyles_lambda`** — rolling OLS of bucket price-change against
   `buy_volume - sell_volume` (the BVC-classified order-flow proxy),
   identical numerically-stabilized formula as crypto's `kyles_lambda`.
4. **`amihud_illiquidity`** — `rolling_mean(|price_change| / volume, 50)`, a
   model-free companion metric (Amihud 2002) that doesn't depend on BVC's
   classification at all — included as a sanity check against `kyles_lambda`:
   if the two diverge sharply, that's a signal to distrust the BVC
   classification for that period rather than trust either number blindly.

Output: `processed/{PAIR}/fx_toxicity/YYYY/MM/DD/`, a **separate** table from
Phase 2's `fx_physics/` (different bucketing axis — volume vs time), validated
against the new `ForexToxicitySchema`.

## How this joins back onto the time-bucketed physics features

Rather than forcing volume-bucketed toxicity onto the time grid inside this
module (which would blur the volume-clock property that makes VPIN meaningful
in the first place), the join is left to `time_alignment.py` (Phase 2):
`fx_toxicity`'s irregular volume-bucket timestamps forward-fill onto the
shared time grid exactly like any other asset stream, gaining an `is_stale`
flag for free. This keeps each module doing one job.

## Stated limitations (not hidden)

- BVC is a **statistical classification, not ground truth** — it will
  misclassify some volume, same as it would for any market lacking real
  trade tags. Treat `bvc_vpin`/`kyles_lambda` as informed estimates, not
  exact figures, the same caveat that applies to the original VPIN paper's
  own back-tests.
- Quoted volume (`bid_volume + ask_volume`) is a proxy for activity, not
  executed size — Dukascopy doesn't publish executed size on the free tick
  feed. This is a real data ceiling, not a modeling choice.
- The BVC price-change std (`rolling_std`) needs a long-enough rolling
  window to be stable; very illiquid pairs/sessions with few ticks per
  bucket will produce noisier `z_t` and therefore noisier classification —
  worth widening `W` or the bucket size for minor pairs rather than reusing
  the majors' defaults blindly.
- **No cross-day volume-bucket remainder carryover.** `feature_engineering.py`
  tracks a leftover-volume remainder across midnight (`_get_previous_remainder`
  / `_save_remainder`, its "Dynamic Lookback" fix) so a volume bucket never
  gets artificially cut short at a UTC day boundary. `forex_toxicity_engineering.py`
  processes one day at a time and does **not** carry that remainder forward
  yet — the last bucket of each day will be a partial bucket. For FX this is
  a smaller issue than for crypto (there's a real ~48h weekend closure
  breaking continuity anyway, and daily UTC boundaries fall mid-session
  rather than at a natural close), but it's a real simplification, not an
  oversight to gloss over — worth porting the same remainder-carry pattern
  if bucket-boundary precision matters for a given use case.
- **`_normal_cdf` uses `np.vectorize(math.erf)`**, which is a Python-level
  loop under the hood, not a true NumPy vectorized ufunc. Fine for one day's
  ticks; if this is ever run over very large tick counts in a hot loop, swap
  in `scipy.special.ndtr` (vectorized normal CDF) or `scipy.special.erf` for
  real vectorization — this was kept dependency-light (stdlib-only) rather
  than pulling in `scipy` for one function.
