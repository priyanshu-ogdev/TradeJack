# Signal Composition Layer

`execution/composition_layer.py` composes the RL policy's action, the supervised
predictor's gated opinion, and `risk_guardian`'s veto into one final order intent.

## v2: dynamic, continuous sizing (greedy AND cautious)

The first version picked one of three flat multipliers (1.0 / 0.5 / 0.0) purely from
which bucket a trade fell into — cautious by construction, but never greedy: a
99%-confidence agreement with an untouched risk budget got exactly the same size as a
56%-confidence agreement one tick above the trust threshold with the risk budget nearly
exhausted. That's not "dynamic," it's a lookup table.

v2 replaces the flat multiplier with **four independently-tunable, multiplicatively-
combined scalars**, each bounded and each independently tested:

1. **Greedy axis — confidence-scaled agreement bonus.** On the `agree` bucket only,
   size scales continuously from `agree_multiplier` (at the trust threshold,
   `min_predictor_confidence`) up to `greedy_ceiling` (at full predictor confidence).
   Real, above-baseline conviction earns a genuinely larger position — this is the
   actual "greedy" half, and it's bounded (`greedy_ceiling`, default 1.5) so greed is
   never unbounded.
2. **Cautious axis 1 — risk-budget throttle.** As the fraction of the daily loss
   budget already used (`risk_budget_used_fraction`, caller-supplied — typically
   derived from `RiskGuardian.state`) climbs past `risk_budget_caution_start`
   (default 0.5), size ramps linearly down toward `risk_budget_min_scalar` (default
   0.15). This is **pre-emptive and graduated**, complementing `risk_guardian`'s
   binary halt rather than replacing it — the desk gets quieter as the day's cushion
   shrinks, instead of trading at full size right up until the hard veto fires.
3. **Cautious axis 2 — toxicity throttle.** If a `bvc_vpin` reading is supplied (see
   `data_forge/forex_toxicity_engineering.py`), size ramps down past
   `vpin_caution_threshold` (default 0.3) toward `vpin_min_scalar` (default 0.3) —
   adverse selection is a real, currently-measurable cost, not a hypothetical one.
4. **Disagreement is inverted, not just floored.** If `disagree_multiplier` is
   configured above its 0.0 default (a strategy may want a small floor instead of an
   outright skip), a *more confident* disagreement shrinks that floor further — never
   grows it. Confidently disagreeing is never a reason to size up.

All four combine multiplicatively, then the result is clipped to
`[0, max_size_fraction]` — a hard cap that's a correctness guarantee independent of how
the four tuning knobs above are set (verified directly: `max_size_fraction` still holds
even when `greedy_ceiling` is deliberately misconfigured above it).

## v3: the fifth axis (performance streaks) + live wiring

v2 flagged a missing fifth axis: a scalar based on *realized* trade outcomes,
independent of the equity-based risk-budget throttle. `execution/performance_tracker.py`
adds it, wired into `SignalComposer.compose()` as `performance_scalar` (same "absence is
neutral, not evidence" convention as `risk_budget_used_fraction`/`bvc_vpin`).

**Deliberately asymmetric, not a symmetrical "momentum of performance" axis**:
- **Loss-streak caution is on by default** — after `loss_streak_caution_start` (2)
  consecutive losses, size ramps down toward `loss_streak_min_scalar` (0.3), holding
  at that floor past `loss_streak_floor_at` (6) losses rather than continuing toward
  zero — the risk-budget throttle and `risk_guardian`'s hard halt are the backstops
  for "stop entirely"; this tracker's job is to lean cautious, not veto.
- **Win-streak "greed" is OFF by default** (`win_boost_ceiling=1.0`). This is a
  genuinely contested idea — "the signal is currently working, lean in" vs. the
  hot-hand fallacy (a short win streak is weak evidence, and sizing up invites a much
  larger loss on reversion). This module takes no position; enabling it is a
  deliberate, informed opt-in, not a default.

`execution/live_composer.py` (`LiveOrderComposer`) closes the "hand-fed numbers" gap
from v2: it binds one `RiskGuardian` + one `SignalComposer` + one
`PerformanceStreakTracker` so `risk_budget_used_fraction` and `performance_scalar` are
pulled from the real objects automatically on every `compose_order()` call, instead of
the caller re-deriving them by hand each time. It adds no new decision logic of its own
— duplicating `RiskGuardian`/`PerformanceStreakTracker`'s own rules here would be
exactly the kind of drift bug already caught once (`risk_guardian.py`'s
`risk_budget_used_fraction()` replaced two separately-inlined copies of the same
formula with one).

`RiskGuardian.risk_budget_used_fraction()` (new) exposes, as a single source of truth,
the same loss-budget formula `update_equity()`/`check()` already used internally for
the hard halt — 0.0 while no loss, ramping toward 1.0 at the daily loss limit, clipped
at 1.0 on a breach.

## v4: wired into the actual inference server

`LivePaperInferenceServer` (`execution/live_inference_server.py`) now has an opt-in
`use_composition_layer=False` constructor flag. When `True`, `_maybe_act()` routes the
RL model's raw action through a bound `LiveOrderComposer` before submission instead of
sending it straight to the exchange. `False` (the default) preserves the exact prior
behavior byte-for-byte.

The nontrivial part is unit reconciliation: `_think()`/`submit_target_position()` work
in fraction-of-equity space (`[-1, 1]`), while `SignalComposer`/`LiveOrderComposer`
work in asset-quantity space (`base_qty`, so the internal `risk_check_fn` can validate
real order cost). The conversion is an exact round-trip, not an approximation:

```
base_qty    = |target_frac| * equity / price          # what RL's raw fraction implies in qty
proposed_qty = base_qty * intent.size_fraction         # done inside compose() itself
final_frac  = intent.direction * |target_frac| * intent.size_fraction
            = proposed_qty * price / equity            # algebraically identical
```

Verified directly, not just derived on paper: ran this round-trip through the real
`SignalComposer.compose()` (no mocks) across four scenarios (positive/negative RL
actions, a deadzone-range action, and a large-fraction/expensive-asset case) and
confirmed `proposed_qty` (what the composer actually validated internally) exactly
equals what `final_frac` implies back in quantity space, bit-for-bit, in every case.

One design decision worth calling out explicitly: a composed `direction == 0` is
**not** always safe to treat as "submit target_frac=0.0" (which would force-close any
existing position). It only means that when the bucket is `"flat_rl"` — i.e. RL's own
raw action was already flat. For every other zero-direction bucket
(`"risk_vetoed"`, `"predictor_locked"`, or a throttle stack that scaled a real RL
intent down to ~0), `_maybe_act()` explicitly **skips submission that tick** instead —
"don't place the trade RL wanted" is not the same instruction as "close whatever
is currently open," and conflating them would mean a risk veto could force a
position closed that the veto was never actually about. Verified this distinction is
real, not just theoretical, by composing with a `risk_check_fn` that always rejects:
the result is `bucket="risk_vetoed"`, `direction=0`, which is a different code path
in `_maybe_act()` from a genuine `bucket="flat_rl"`, `direction=0`.

Also fixed while building this (an independent bug, not specific to the live wiring):
`SignalComposer._predictor_opinion()` had no guard around the actual sklearn
`predict()`/`predict_proba()` calls — any shape or dtype mismatch between a supplied
`feature_window` and what a predictor's model was fit on raised a hard, unhandled
`ValueError` straight out of `compose()`. Reproduced the exact crash with a fake
predictor before fixing it: now any exception there degrades to "no usable opinion
this cycle" (the same safe path already used for "not enough data"), not a crash.

## Order of operations (unchanged from v1, still the load-bearing part)

1. **RL proposes.** Flat RL action (inside `direction_deadzone`) → `bucket="flat_rl"`,
   no trade, full stop — none of the four scalars above get a chance to manufacture one.
2. **Predictor + risk/toxicity scale.** As described above.
3. **`predictor_locked` interlock.** During a `PlasticityManager` reset, no order is
   proposed regardless of any scalar.
4. **`risk_guardian` vetoes last**, against the actual already-scaled `(side, qty,
   price)` — a veto zeroes the trade even after a favorable greedy scale-up.

## Why these defaults, and why they're not final

Every number here (`agree_multiplier=1.0`, `greedy_ceiling=1.5`,
`unconfirmed_multiplier=0.5`, `disagree_multiplier=0.0`, `risk_budget_caution_start=0.5`,
`risk_budget_min_scalar=0.15`, `vpin_caution_threshold=0.3`, `vpin_min_scalar=0.3`,
`max_size_fraction=1.5`) is a **stated, deliberately conservative starting point — not a
backtested-optimal setting**. `greedy_ceiling == agree_multiplier` disables the greedy
axis entirely and recovers the old fixed-1.0x behavior, if a deployment wants that.

## What this does NOT do

- It does not convert `rl_action`/`size_fraction` into an actual `qty` itself — that's
  the caller's own position-sizing logic (`base_qty` is passed in already computed).
  See "v4: wired into the actual inference server" above for exactly how
  `live_inference_server.py` does this conversion in both directions.
- It does not call `risk_guardian` itself if `risk_check_fn` is `None` — intentional
  for unit-testing the composer in isolation; a live caller **must** always pass a
  real `risk_check_fn`.
- It does not compute `risk_budget_used_fraction` itself when called directly
  (`SignalComposer.compose()`) — that's a caller-supplied reading; absence is treated
  as neutral (scalar 1.0), not as evidence of elevated risk. `LiveOrderComposer`
  removes this by pulling it from a bound `RiskGuardian` automatically. `bvc_vpin`
  works the same way, with one more option: pass `toxicity_symbol=` to
  `LiveOrderComposer` and it auto-loads the latest reading via
  `TrainingTableBuilder.latest_toxicity_reading()` (cached with a TTL) — an explicit
  `bvc_vpin=` on a given call always overrides the auto-loaded value.
- It does not manage the `predictor_locked` flag's lifecycle — that's
  `PlasticityManager`'s / the training-loop orchestrator's job.
- `LiveOrderComposer.reset_for_new_session()` only clears the performance-streak
  tracker — it deliberately does NOT call `risk_guardian.reset_halt()`, which stays a
  separate, human-reviewed action per `docs/PROCESS_SUPERVISION.md`'s "risk halt !=
  process crash" principle.

## Verification status

Real execution, not tracing:
- `tests/test_composition_layer.py` — 17 tests (14 from v2 + 3 for the v3
  `performance_scalar` axis), all passing via
  `python -m unittest tests/test_composition_layer.py -v`.
- `tests/test_performance_tracker.py` — 9 tests, all passing, covering the loss-streak
  ramp-and-hold-at-floor behavior, the win-streak boost being off by default and
  correctly enabled when opted in, and streak-reset semantics (scratch trades, sign
  flips, manual reset).
- `tests/test_live_composer.py` — 5 tests, full-stack: a genuine `RiskGuardian`, a
  genuine trained `SignalPredictor`, and a genuine `PerformanceStreakTracker` composed
  together with no mocks. Proves a real `update_equity()` loss call actually throttles
  the *next* `compose_order()` call, a real daily-loss breach produces a real
  `risk_vetoed` intent (not a simulated one), and `record_fill()` genuinely feeds the
  performance tracker into the following order.
- `execution/risk_guardian.py`'s own embedded self-test (`python
  execution/risk_guardian.py`) still passes unchanged after adding
  `risk_budget_used_fraction()` — re-run directly, not assumed unaffected.
- **v4 (live wiring)**: the fraction<->quantity round-trip formula was verified
  directly against the real `SignalComposer.compose()` (no mocks) across four
  scenarios, confirming `proposed_qty` (what the composer actually validates
  internally) exactly equals what `final_frac` implies back in quantity space in
  every case. The `risk_vetoed`-vs-`flat_rl` skip/flatten distinction was verified
  with a `risk_check_fn` that always rejects, confirming the two zero-direction
  cases are genuinely distinguishable in `_maybe_act()`, not just in comment text.
  The `_predictor_opinion()` shape-mismatch fix was verified by reproducing the
  actual unguarded crash with a fake predictor first, then confirming the fix
  survives it and falls back to `"unconfirmed"`. Full regression: all 48 existing
  tests across `test_composition_layer.py`, `test_performance_tracker.py`,
  `test_live_composer.py`, and `test_supervised_predictor.py` still pass after
  these changes. What is **not** verified here, and can't be from this sandbox
  (no `torch`/`gymnasium`/`stable_baselines3`): `_maybe_act()` itself end-to-end
  against a real trained RL model and a real depth feed — the composition math and
  the composer's own logic are proven; the full asyncio decision loop wiring it
  sits inside is verified by direct source reading and syntax-checking only.
