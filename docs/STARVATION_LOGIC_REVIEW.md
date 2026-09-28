# Starvation/Survival-Tax Logic — Review

Scope: `warden/warden_core.py`'s `WardenHypervisor.apply_survival_tax()` and
`physics/portfolio_tracker.py`'s `PortfolioAccountingEngine` — the legacy `--legacy`
swarm path's "cost of staying alive" economic model (the mechanism referred to
early in this project as "starving the agent to deploy more and account for more
scenarios").

## What the formula actually does

Two additive components:
1. **Base log tax**: `base_tax_per_hr * (1 + alpha * log(1 + simulated_hours))` —
   scales up slowly (logarithmically) the longer a child has been alive.
2. **Stagnation penalty**: kicks in once a child has gone more than 4 real/market
   hours without a new equity high-water-mark, scaling linearly with how far past
   that threshold it's gone.

## The finding

Traced `simulated_hours`'s source precisely rather than accepting the formula at
face value: it used to be `summary.ticks_active / 60.0` — an implicit assumption
that exactly 60 ticks equal one hour, i.e. one tick = one real/market minute.

That assumption is **false** for the data this actually runs on. Confirmed by
reading the real pipeline, not assumed:
- `data_forge/feature_engineering.py`'s `TradeFlowPhysics` buckets by **volume**
  (`volume_bucket_size`), not fixed time intervals.
- `physics/lob_env.py`'s `TradeJackLOBEnv` — the exact environment
  `swarm/child_agent.py` uses (confirmed directly: `from physics.lob_env import
  TradeJackLOBEnv`) — increments `ticks_active` once per volume bucket processed,
  not once per real/market minute.

So one tick can span a few seconds (a high-volume regime — exactly the
flash-crash/liquidity-vacuum synthetic scenarios this project trains against, see
`docs/PHASE3_SCENARIO_DIVERSITY.md`) or several hours (a quiet market). The old
formula was blind to this: it would tax a child based on how many volume buckets
happened to fill, not how much real time it had actually been alive.

**Concretely verified, not just derived on paper**: ran two real scenarios through
the actual `PortfolioAccountingEngine`/`WardenHypervisor` — 120 ticks compressed
into 10 real minutes (high volume) vs. the same 120 ticks spread across 10 real
hours (low volume). The old formula (`ticks_active/60`) would compute the exact
same `simulated_hours=2.0` for both — completely blind to the 60× difference in
actual elapsed time. This inverts the tax's own intent in exactly the highest-
stakes case: a child would be taxed *more* for successfully surviving a volatile,
high-activity period (when it's arguably doing the most valuable thing this
whole project cares about) than for sitting through a long, uneventful stretch.

One related, deeper finding flagged but deliberately **not** fixed in this pass
(out of scope — would mean re-deriving Sharpe/Sortino's own math, not just the
tax): `PortfolioAccountingEngine.__init__`'s `ticks_per_year = 365.0 * 1440.0`
(1440 = minutes/day) bakes the same "1 tick = 1 minute" assumption into the
Sharpe/Sortino annualization factor used for tier assignment and survival-mode
decisions. Same root cause, larger blast radius — worth its own dedicated review
if this legacy path is ever revived in earnest.

## The fix

Reused the exact methodology already proven correct for the stagnation-penalty
half of the same formula: a **real market-timestamp delta**, not a tick-count
proxy.

- `PortfolioAccountingEngine` now tracks `first_market_timestamp` — the real
  `market_timestamp` of a child's very first `record_step()` call, set once,
  mirroring exactly how `last_hwm_market_timestamp` is already set on that same
  first call.
- `apply_survival_tax()` now computes
  `simulated_hours = (market_timestamp - first_market_timestamp) / 3600.0` —
  genuine elapsed real/market time, immune to volume-bucket density.
- Both halves of the tax formula are now internally consistent: real elapsed
  time, not a tick-count proxy for one half and a real timestamp for the other.

## Two more real bugs found while verifying the fix, not assumed away

1. **A second, independently-drifted copy of the schema.** `WardenHypervisor.
   init_child_ledger()` maintains its own `CREATE TABLE portfolio_state` and seed
   `INSERT`, completely separate from `PortfolioAccountingEngine._init_sqlite()`.
   Only found this by actually running the existing test suite
   (`tests/test_warden_hardware.py`) after the first version of the fix — it
   failed with `AttributeError: 'NoneType' object has no attribute 'is_alive'`,
   tracing back to `audit_child_ledger()` reading a row with a mismatched column
   count against the new field. Fixed both schema sites identically (new column +
   `ALTER TABLE` migration + updated seed `INSERT`), and this duplication itself
   is worth naming as a standing risk: two schema definitions for the same table,
   maintained separately, will drift again the next time either one changes
   without the other being updated in the same pass.

2. **A resumed/migrated child could get a catastrophic phantom tax bill.**
   `PortfolioAccountingEngine` doesn't hydrate `ticks_active`/`first_market_
   timestamp` from an existing ledger at construction — a genuinely old ledger,
   once migrated (`ALTER TABLE ... ADD COLUMN ... DEFAULT 0.0`), would have
   `first_market_timestamp=0.0` until that process's own first `record_step()`
   call. `apply_survival_tax()` would then compute elapsed time since the Unix
   epoch — billions of seconds — as this child's "hours alive," instantly
   bankrupting it via the log tax. Fixed with an explicit guard: `first_market_
   timestamp <= 0.0` skips the base tax for that cycle (the stagnation penalty,
   which doesn't depend on it, still applies normally) rather than computing a
   nonsensical value. Verified this guard fires correctly on a freshly-migrated
   ledger, and verified it genuinely self-heals — the very next real
   `record_step()` call sets a proper value and the tax returns to normal on the
   next audit cycle. Neither of these was a hypothetical concern by the time I
   found them: both were reproduced with real SQLite ledgers and real
   `WardenHypervisor` calls, not inferred from reading the code.

## Verification status

Unlike most of this project's `data_forge`/`physics`/`training` modules,
`physics/portfolio_tracker.py` and `warden/warden_core.py` need only `numpy` and
the standard library — **no torch, gymnasium, or pydantic_settings** — so this
was genuinely executable in this sandbox, and was executed:
- The core units bug: reproduced with two real scenarios through the actual
  classes (not mocks), confirming the old formula's blindness and the new
  formula's correct sensitivity to real elapsed time.
- The schema-migration path: exercised against a hand-built, genuinely
  old-schema SQLite ledger, confirming the `ALTER TABLE` migration adds the
  column correctly and doesn't corrupt existing rows.
- The phantom-tax edge case: reproduced end-to-end across a simulated process
  restart, confirming both the guard's protection and its self-healing behavior.
- Full existing test suite (`tests/test_warden_hardware.py`,
  `tests/test_evolution_state.py`): all pass after the fix.

What's still unverified: whether this legacy `--legacy` swarm path is ever
actually exercised by a real training run in your environment — this review
covers correctness of the tax formula itself, not whether the legacy path as a
whole is worth reviving over the active SB3 pipeline this project has been built
around throughout every other phase of this work.
