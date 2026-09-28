# TradeJack Control Panel

A live control surface for the trading system: equity, risk/halt status, pending
model promotions, and recent trades, plus start/stop trading, start/stop continuous
training, and reset-halt controls.

## Architecture

```
┌─────────────────────┐     REST (status, trades,      ┌──────────────────────┐
│  React frontend      │     promotions, controls)       │  FastAPI backend      │
│  (this directory)    │ ───────────────────────────────▶│  control_panel_api.py │
│                      │                                  │                       │
│                      │◀──── WebSocket (/ws/live,        │  wraps the real:      │
│                      │      1 status frame/sec)          │  - LivePaperInference │
└─────────────────────┘                                   │    Server             │
                                                            │  - ContinuousTrainer  │
                                                            │  - RiskGuardian       │
                                                            └──────────────────────┘
```

The frontend adds no trading logic of its own — every number and every action maps
directly to a method or attribute on the real Python objects
(`execution/live_inference_server.py`, `training/continuous_trainer.py`,
`execution/risk_guardian.py`) that this whole project has already built and tested.
This is a control surface over that system, not a reimplementation of it.

## Running it

**Backend** (from the `TradeJack/` project root):

```bash
pip install fastapi "uvicorn[standard]"
export TRADEJACK_API_TOKEN="pick-a-real-secret"     # required -- refuses to start without it
export TRADEJACK_FRONTEND_ORIGIN="http://localhost:5173"
uvicorn execution.control_panel_api:app --host 0.0.0.0 --port 8000
```

**Frontend**:

```bash
cd frontend/control-panel
npm install
cp .env.example .env.local     # then set VITE_API_TOKEN to the same value as above
npm run dev
```

Open the printed `localhost:5173` URL.

## Design

Built as an instrument panel, not a SaaS dashboard, on purpose — see the panel's
own visual language: a persistent status bar, equity and risk get the most visual
weight, and the Controls section is a physically separate, higher-friction rail
(every consequential action — start trading, stop trading, reset a halt, approve
a promotion — goes through a confirmation dialog) rather than blending in with
passive telemetry. Color is semantic only: teal for nominal, amber for caution,
red for halted — it never changes for decoration.

## Verification status — read this before trusting it

This was built and reviewed in a sandbox with **no network access**, so:

- `npm install` was never run here (the npm registry itself was unreachable —
  confirmed directly, not assumed). **You must run it yourself before this
  will actually start.**
- Every `.jsx`/`.js` file was, however, verified for real: each file was
  transpiled individually with a real `esbuild` binary (bundled with a
  different globally-installed tool, found and reused rather than skipping
  verification) and confirmed to produce zero syntax errors, and then the
  entire app was bundled from `src/main.jsx` through every relative import —
  all 9 components plus both `lib/` modules plus the Tailwind CSS — with
  **zero errors and zero warnings**. This confirms every import path resolves
  and every file is syntactically valid JSX/JS; it does **not** confirm the
  app renders correctly in a real browser, since that needs a running dev
  server and an actual backend to talk to, neither of which existed in this
  sandbox.
- `execution/control_panel_api.py` (the FastAPI backend) could not be run
  either — `fastapi`/`uvicorn` aren't installed and pip has no network access
  here. It was written by directly reading and matching the exact method and
  attribute names of the real objects it wraps (not assumed from memory), and
  checked with `python -m py_compile`, but has never received a real HTTP
  request or WebSocket connection. In particular, verify the WebSocket
  broadcast loop's behavior under a real client disconnect yourself — that's
  the single thing in this file most likely to have a subtle bug that static
  review can't catch.
- Before trusting the Start Trading / Reset Halt buttons against a real
  account: test against `exchange_mode="paper"` first, exactly like every
  other part of this project's own testnet-before-live discipline.
