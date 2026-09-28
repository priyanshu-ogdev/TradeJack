# Full Design Review — Architecture Correction + Status

## The headline finding: I built redundant, worse-coupled infrastructure without checking for existing infrastructure first

Reviewing "the entire design" surfaced something the narrower per-turn reviews
couldn't: `dashboard/telemetry_server.py` — a complete, already-correct, already
decoupled Flask monitoring dashboard — already existed in this repo when I built
`execution/control_panel_api.py` (FastAPI) plus a full React frontend for the same
purpose. I built it without first checking whether something like it already
existed. That's a real process failure, not just duplicated effort, and it's worth
naming plainly rather than glossing over.

It's worse than simple duplication on one specific axis. `telemetry_server.py`'s
own `/api/kill-switch` endpoint states its guiding principle directly: *"This
dashboard doesn't need its own halt mechanism — it just writes/removes the same
file the trading loop was already built to check."* That's the same principle
`execution/risk_guardian.py`'s `kill_switch_path` and
`training/continuous_trainer.py`'s `pending_promotions/*.json` files already
established, twice over, before I ever touched this. `control_panel_api.py` went
the opposite direction: an in-process `AppState` holding a live reference to
`LivePaperInferenceServer`/`ContinuousTrainer`, started via
`asyncio.create_task()` *inside the API process itself* — meaning the control
surface and the trading process became the same process. A crash or restart of
one could now affect the other, exactly what the existing pattern was built to
prevent.

A related, more precise finding along the way: `telemetry_server.py` reads from
`state/child_{account_id}/...` — the same per-account directory convention
`execution/paper_exchange.py`'s `fill_db_path` actually uses (confirmed by reading
both sides, not assumed). So this wasn't "two unrelated dashboards for two
different subsystems" — it was one already-correct, already-general dashboard I
failed to discover, covering exactly the case I thought needed a new one.

## The correction, made this pass

- **`execution/command_channel.py`** (new): a small, generic file-based command
  queue — `issue_command()` writes a command file (atomic write-then-rename, same
  pattern as this project's other durable file state), `poll_and_execute()` reads
  and dispatches pending commands to registered handlers, removing each file
  whether its handler succeeds or raises (a broken command can never wedge the
  queue, and can never crash the poller's own loop). Verified directly: issuing
  and consuming commands, an unhandled command type being discarded with a
  warning rather than stuck forever, a raising handler still consuming its file,
  and payload passthrough for `approve_promotion`'s `agent_id` — all four
  confirmed by real execution.
- **`dashboard/telemetry_server.py`** now has the missing write routes
  (`/api/trading/{start,stop}`, `/api/training/{start,stop}`,
  `/api/risk/halt/reset`, `/api/promotions/<id>/approve`), all just calling
  `issue_command()` — this process still never touches a live trading object
  directly, consistent with every other route it already had.
- **`scripts/genesis_prime.py`**'s `run_live()` now polls the command channel
  once every 2 seconds alongside its existing graceful-shutdown task, wired to
  `server.stop()` / `trainer.stop()` / `risk.reset_halt()` /
  `trainer.approve_pending_promotion()`. `START_TRADING`/`START_TRAINING` are
  deliberately *not* handled here — a stopped process can't poll for its own
  start signal; those two commands are for `scripts/process_supervisor.py` (or
  an operator) to consume by actually spawning the process, and this distinction
  is documented in `command_channel.py` rather than left implicit.
- **`execution/control_panel_api.py`** is now clearly marked deprecated at the
  top of the file, with the reasoning and the migration path stated directly,
  rather than silently left to rot or deleted outright (in case the async/
  WebSocket shape is useful for something else later).

## Honest gap left open, not rushed

The `frontend/control-panel/` React app still talks to the deprecated API's
endpoint shapes. Re-pointing it at `telemetry_server.py`'s actual response shapes
(`/api/status`'s per-account array instead of a single merged object,
`/api/equity-curve` instead of `/api/equity/history`, etc.) is a small,
mechanical change to `src/lib/api.js`, but I'm not rushing it in the same pass as
finding and fixing the underlying architecture mistake — verifying it correctly
against the *real* endpoint shapes (not guessed ones, the same discipline this
whole review is about) is the honest next step, not something to paper over with
an unverified edit under time pressure.

## Where the system actually stands now

**Backend, verified and solid:**
- Phase 0–1 safety rails: promotion gate, synthetic-feed decoupling, risk-guardian
  equity fix, testnet-before-live gate, `auto_promotion` gate, process supervisor
  with real graceful shutdown (SIGTERM fix), sticky-halt-plus-alerting.
- Phase 2: live trade capture → `TradeFlowPhysics` bridge, closing the
  "nothing live ever reaches retraining" gap, plus a real (not theoretical)
  non-determinism fix in `process_daily_file()`'s file selection.
- Phase 3: literature-grounded scenario diversity (flash crash / melt-up /
  two-sided / liquidity vacuum / spoofing-via-decoupled-OFI-spike), all verified
  by direct execution, not just designed.
- Composition layer + portfolio allocator: signal confirmation, risk-budget/
  toxicity/performance throttles, two-slot correlation-aware allocation — 124+
  tests passing for real.
- The starvation-tax units bug (volume-bucketed ticks treated as fixed-time
  ticks) and the observation-normalization train/serve skew (fresh per-window
  z-score vs. the EWMA the model was actually trained against) — both real,
  both verified bit-for-bit or numerically against the actual training-side code,
  not just plausible-sounding.

**Now corrected this pass:** the control-surface architecture itself.

**Still open, honestly:**
- Frontend re-pointing (above).
- GAN-spoofer/HER porting from the legacy swarm path into the active SB3
  pipeline (Phase 3 continuation) — deliberately scoped out previously since it
  touches the core RL env/replay buffer, untestable in this sandbox.
- Phase 4: statistical power check on the airgap's splits,
  `ewc_sb3_adapter`/`plasticity_manager` test-coverage gap.
- Everything that has ever required `torch`/`gymnasium`/`stable_baselines3`/
  network access in this entire project: none of it has been run end-to-end.
  Every fix in this codebase has been verified as rigorously as a sandbox with
  no network and a partial Python dependency set allows — direct execution
  wherever the code was pure-enough to run, exact interface-matching and
  `py_compile` everywhere else — but a real environment, a real testnet
  connection, and a real trained model are the one verification step nothing
  here can substitute for.
