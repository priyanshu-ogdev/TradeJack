# Phase 3: Scenario Diversity (first slice) — "more scenarios, more resilience,
where it actually reaches the deployed model"

## Phase 2 review (done first, per instruction)

Before starting Phase 3, reviewed the Phase 2 work rather than assuming it was
correct. Found one real bug, more serious than how I'd originally documented it:

**`feature_engineering.py`'s `process_daily_file()` had no explicit sort on its
glob results.** `glob.glob()`'s return order is OS/filesystem-dependent, not
alphabetical — so if both a genuine bulk-downloaded file and `live_trade_bridge.py`'s
bridged output existed for the same date, which one got processed into
`physics.parquet` was **non-deterministic across runs/environments**, not just "an
arbitrary but fixed choice" as the first version of the Phase 2 doc claimed. Fixed:
file selection is now sorted deterministically, with a genuine bulk-download file
always preferred over a live-bridged one when both exist. Verified with real string
fixtures in both input orders, confirming the same file wins regardless of glob's
original (arbitrary) ordering. Both `live_trade_bridge.py`'s and
`PHASE2_LIVE_LEARNING_LOOP.md`'s docstrings/caveats were corrected to state this
precisely rather than understate it.

## What was actually orphaned, and why

`data_forge/synthetic_diffusion.py`'s `SyntheticDiffusionEngine.generate_black_swan_scenario()`
was fully built but never called by the training pipeline. Traced the actual root
cause rather than just "nobody wired it up yet": it writes to
`data_store/synthetic/{symbol}/{name}.parquet`, but `TradeJackLOBEnv` — the
environment every active training path uses — only ever scans
`data_store/processed/{symbol}/physics/`, a completely different, unscanned
directory. Even if something had called it, the output would have landed
somewhere nothing reads.

Two ways to fix that: teach `TradeJackLOBEnv` to also scan `data_store/synthetic/`
(touches the core RL env, untestable here without `gymnasium`/`torch`, and would
change behavior for every training path, not just the tournament), or materialize
each scenario directly into the directory that's already scanned, at a clearly
reserved, synthetic date. Took the second, minimal-blast-radius option — consistent
with every other fix in this project preferring to extend what's already trusted
and scanned over widening a shared core component.

## A second real gap, found while reading the function, not assumed away

Despite the upgrade plan describing this generator as supporting "configurable
volatility multipliers, named regimes," the actual stochastic process only ever
varied by `volatility_multiplier` — jump direction was hardcoded negative. Every
"regime" was really the same one-sided-crash pattern at different intensities, not
a structurally different tail event. A melt-up squeeze and a flash crash are
economically very different things to train resilience against; the generator
couldn't produce the former at all.

Fixed by splitting the pure-numpy generation out of the disk-writing (`SyntheticDiffusionEngine._generate_scenario_arrays()`,
zero I/O, directly reusable) and adding two new optional parameters,
`jump_direction` (-1 crash / +1 squeeze / 0 two-sided) and `jump_probability`,
both defaulting to the exact prior hardcoded values — fully backward compatible
(confirmed genuinely orphaned first: grepped the whole repo, zero other callers,
zero tests, so there was nothing to actually break either way).

**Verified by real execution, not just derived on paper:** ran the new generation
logic directly with `jump_direction=-1` (confirmed net price decline),
`jump_direction=+1` (confirmed net price incline), and `jump_direction=0` over a
larger sample (confirmed jumps split roughly evenly between positive and
negative, not all one-sided) — and confirmed the default-argument case reproduces
the exact same price path as an explicit `jump_direction=-1.0` call, proving
nothing changed for anyone relying on the old behavior.

## What was built

- **`training/scenario_injection.py`** (new): `inject_synthetic_scenarios(symbol, ...)`
  writes four structurally different regimes —
  `flash_crash`, `melt_up_squeeze`, `two_sided_high_vol`, `liquidity_vacuum`
  (approximated via a rare-but-large-jump parameterization, since this generator's
  output columns have no direct bid-ask spread field to model book thinness more
  literally) — as `data_store/processed/{symbol}/physics/2099/01/{01-04}/physics.parquet`.
  `2099` is chosen to be unambiguously synthetic and trivially
  greppable/removable as one block (`remove_synthetic_scenarios()` is the
  matching cleanup function).
- **Idempotent by default**: a regime already present on disk is skipped, not
  regenerated, unless `force=True`. Deliberate — these are meant to be a stable,
  repeatable stress-test fixture an agent sees roughly the same way cycle over
  cycle, not something that silently reshuffles on every tournament construction.
  Verified directly: first call generates all four, a second call with the same
  fixtures present skips all four, `force=True` regenerates all four regardless,
  and two different symbols don't collide with each other's reserved block.
- **Wired into `CrucibleTournament.__init__`** via `inject_synthetic_scenarios=True`
  (opt-in, defaulting on). Wrapped in `try/except` on purpose, and verified for
  real in this sandbox's actual missing-dependency failure mode: a broken
  injection (missing `polars`, a permissions issue) logs a warning and lets
  tournament construction continue, rather than blocking it.

## What this deliberately does NOT do yet

The original Phase 3 scope also named two pieces of the "diverse scenario"
toolkit that exist only on the deprecated legacy path (`swarm/child_agent.py`,
the `--legacy` 50-container swarm) and were never ported to the active SB3
pipeline:

- **`AdversarialGANSpoofer`** (order-book spoofing injection for manipulation
  robustness) — would need porting into `physics/lob_env.py` as an
  observation-augmentation option.
- **`HindsightExperienceReplay`** — the active SB3 path uses a different
  prioritized-replay implementation (`swarm/replay_buffer.py`/
  `swarm/sb3_replay_buffer_adapter.py`) with no HER goal-relabeling; porting real
  HER in would mean changing the actual replay buffer's sampling logic.

Both are meaningfully larger, riskier changes than this pass — they touch the
core RL environment and replay buffer respectively, neither of which is
executable or testable in this sandbox (`gymnasium`/`torch`/`stable_baselines3`
all unavailable), unlike the pure-filesystem/pure-numpy work in this pass, which
was directly verified by running it. Scoped out on purpose rather than attempted
without the ability to verify them — a reasonable next slice of Phase 3, not
something to build blind.

## Verification status

- `_generate_scenario_arrays()`'s regime-shape logic: executed directly, all four
  claims (crash direction, squeeze direction, two-sided mix, default-args
  equivalence) confirmed against real numpy output.
- `inject_synthetic_scenarios()`'s idempotency/force/per-symbol-isolation logic:
  executed directly against a real temp filesystem, all four scenarios passed.
- `CrucibleTournament`'s injection resilience: reproduced the actual
  missing-dependency failure this sandbox has and confirmed it degrades to a
  logged warning, not a crash — the real failure mode, not a theoretical one.
- Full existing 48-test suite, `risk_guardian.py`'s self-test, and
  `live_exchange_bridge`/`deploy_config`'s self-tests: still green after these
  changes.
- **Not verified here**: the actual Parquet-writing path inside
  `inject_synthetic_scenarios()` (needs `polars`), and — most importantly —
  whether `TradeJackLOBEnv` actually picks up and trains on the `2099/01/`
  partitions the way intended (needs `gymnasium`/`torch`/a real training run).
  Run a short real tournament cycle and confirm the synthetic days show up in
  its data rotation before trusting this against real training time.
