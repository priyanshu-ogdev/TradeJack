# Process Supervision for Live/Testnet Trading

Closes the Phase 1 upgrade-plan item: previously, nothing restarted the trading
process if it exited — an unhandled exception, the feed giving up after repeated
failed resyncs, an OOM kill, anything — and there was zero code or docs covering
this for the active `genesis_prime.py --mode {paper,testnet,live}` pipeline.

## Prerequisite fix (already applied): graceful shutdown actually works now

Before any supervisor is safe to use, the supervised process has to actually shut
down cleanly when asked. `genesis_prime.py`'s live/testnet/paper branch used to rely
on `except KeyboardInterrupt` alone — that only fires on `SIGINT` (Ctrl+C). Python's
default handling of `SIGTERM` (what `systemd stop`, `docker stop`, and
`scripts/process_supervisor.py` all send for a graceful stop) is immediate
termination — it does **not** raise `KeyboardInterrupt` and does **not** run
`server.stop()` / `trainer.stop()`. In live/testnet mode that meant open positions,
`RiskGuardian` state, and the exchange connection could all be torn down mid-operation
instead of closed cleanly.

This is fixed: `run_live()` now installs a handler for both `SIGINT` and `SIGTERM`
that triggers the same graceful `server.stop()` / `trainer.stop()` path either way,
and a crashing task's exception now propagates out (non-zero exit) instead of being
silently absorbed — which is exactly what a supervisor needs to tell a real crash
apart from a clean, intentional stop.

## Two layers, and which to use when

**Prefer OS-level supervision (systemd or Docker) whenever it's available.** They run
as PID 1 or a system service and survive `scripts/process_supervisor.py`'s own
process dying — this script cannot supervise itself. Use `scripts/process_supervisor.py`
when neither is available (a bare VM without systemd, local development, Windows).
They compose fine together, but there's rarely a reason to stack both — pick one.

### Option A — systemd (Linux servers with systemd, e.g. the target DGX/cloud deploy)

Install `deploy/systemd/tradejack-live.service` (adjust `User`, `WorkingDirectory`,
and the `--mode`/`--symbol`/`--capital` flags for your setup), then:

```bash
sudo cp deploy/systemd/tradejack-live.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now tradejack-live
sudo systemctl status tradejack-live
journalctl -u tradejack-live -f
```

`systemctl stop tradejack-live` sends `SIGTERM` — with the prerequisite fix above,
that now actually triggers the graceful shutdown path.

### Option B — Docker

```bash
docker run -d --name tradejack-live --restart unless-stopped \
  -v $(pwd)/state:/workspace/state -v $(pwd)/data_store:/workspace/data_store \
  tradejack:latest python -m scripts.genesis_prime --mode live --symbol BTC-USDT --capital 100.0
```

`docker stop` sends `SIGTERM` (after a grace period, then `SIGKILL`) — same
prerequisite fix applies. `--restart unless-stopped` handles the restart-on-crash
job; Docker does not implement backoff or a circuit breaker on its own, so for a
**live** deployment specifically, consider running `scripts/process_supervisor.py`
*inside* the container instead of running `genesis_prime.py` directly, to get the
backoff + circuit-breaker behavior described below on top of Docker's restart policy.

### Option C — `scripts/process_supervisor.py` (portable fallback)

```bash
python -m scripts.process_supervisor --mode testnet --symbol BTC-USDT --capital 100.0
python -m scripts.process_supervisor --mode live --symbol BTC-USDT --capital 100.0 \
    --max-restarts 5 --restart-window-minutes 60
```

Behavior, and why it's not just "restart immediately on exit":

- **Exponential backoff between restarts** (`--initial-backoff-sec`, default 5s,
  doubling up to `--max-backoff-sec`, default 300s). A crash loop that restarts a
  real exchange connection every second is a worse failure mode than one that isn't
  running at all — rapid reconnect/reorder attempts can trip exchange rate limits or,
  worse, race a half-completed order reconciliation.
- **Circuit breaker**: `--max-restarts` (default 5) within a `--restart-window-minutes`
  (default 60) rolling window. Once tripped, the supervisor writes
  `state/supervisor/CIRCUIT_BREAKER_OPEN.json` and **stops trying** — it will refuse
  to even start again while that file exists. This is deliberate: an infinite restart
  loop that keeps quietly failing turns "silently stopped, discover it three days
  later" (the exact failure mode this item exists to close) into "silently
  crash-looping against a live exchange for three days," which is not an improvement.
  A human must read `state/supervisor/restart_log.jsonl` (one JSON line per restart:
  timestamp, exit code, backoff used), investigate, and delete the alert file to
  resume.
- **Graceful shutdown, not reentrant `Popen.wait()`.** Sending the supervisor itself
  `SIGINT`/`SIGTERM` forwards `SIGTERM` to the child and waits up to
  `--graceful-shutdown-timeout-sec` (default 30s) before escalating to `SIGKILL`.
  Note for anyone modifying this file: the main loop polls (`Popen.poll()`) rather
  than blocking on `Popen.wait()` with no timeout — calling `.wait()` from within the
  signal handler while the main loop is *also* blocked inside `.wait()` on the same
  `Popen` object reliably hangs (reproduced directly while building this; not a
  theoretical concern). Both the main loop and the signal handler exclusively use the
  `_poll_wait()` helper for this reason.
- **A clean exit (code 0) is not treated as a crash.** If the supervised process
  exits with code 0 (an intentional, operator-triggered stop of the child itself,
  independent of the supervisor), the supervisor exits too rather than restarting
  something that was deliberately stopped.

## Verification status

The signal-handling fix in `genesis_prime.py` (SIGTERM → graceful shutdown, crash →
propagated exception) was tested directly with fake `server`/`trainer` objects and
real OS signals (`os.kill` with `SIGTERM`) in an isolated asyncio harness — both the
graceful-shutdown and crash-propagation paths were exercised and passed.

`scripts/process_supervisor.py`'s core behaviors were tested against a real
controllable subprocess, with actual OS signals sent across real process boundaries:
circuit breaker tripping after `max_restarts`, refusal to start with an existing
alert file, clean-exit (code 0) not triggering a restart, and — after catching and
fixing a real reentrant-`Popen.wait()` hang during this process — graceful `SIGTERM`
propagation from supervisor to child. The `SIGKILL` escalation path (child ignores
`SIGTERM`) was **not** directly exercised with a genuinely signal-blocking child
process; it reuses the same `_poll_wait()` primitive already proven correct in the
passing graceful-shutdown test, so it's verified by code review and reuse, not by
an independent execution of that exact branch — worth a real test with a
signal-blocking child before depending on it in production.

Neither `deploy/systemd/tradejack-live.service` nor the Docker command above could be
executed in this sandbox (no systemd, no Docker daemon available) — review them
against your actual deployment target before use.
