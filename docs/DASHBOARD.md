# TradeJack Telemetry Dashboard

`dashboard/telemetry_server.py` is a Flask app, run as a **completely
separate process** from the trading loop, that reads directly from the
SQLite ledgers under `state_dir` — the same files
`execution/paper_exchange.py`, `execution/live_inference_server.py`, and
`swarm/training_progress_ledger.py` already write. Nothing here holds an
in-process reference to a running `LivePaperInferenceServer` or
`MultiSleeveOrchestrator`; it doesn't need to. Restarting the dashboard never
affects the trading loop, and vice versa.

## Running it

```bash
export TRADEJACK_STATE_DIR="state"        # wherever your trading loop's state_dir points
export DASHBOARD_TOKEN="pick-a-real-secret"  # required for kill-switch / deposit-address actions
python -m dashboard.telemetry_server
```

Open `http://127.0.0.1:5000`. If you don't set `DASHBOARD_TOKEN`, one is
generated and printed once to the console at startup — the browser prompts
for it the first time you click an action button, then remembers it in
`localStorage`.

## What it shows

- **Account cards** — every `state/child_{id}/` found on disk, with live
  equity/cash/tick.
- **Price & entry/exit chart** — a real price line (sampled at most once/sec
  from live depth updates, a new `price_samples` table added this pass — see
  below) with green ▲ (buy/entry) and red ▲ (sell/exit) markers at actual
  fill prices and times.
- **Equity curve** — from the same portfolio ledger `session_report.py` reads.
- **RL agent progress** — Sortino and equity per training cycle, per agent,
  from the new `swarm/training_progress_ledger.py`, with promotion and
  plasticity-reset events listed alongside.
- **Recent risk decisions** — every approved/rejected decision with its
  reason, so "why isn't it trading" has an actual answer on screen instead of
  needing to grep logs.
- **Binance account** — real balance (if `BINANCE_API_KEY` +
  `BINANCE_ED25519_PRIVATE_KEY_PATH`/`BINANCE_API_SECRET` are configured) and,
  on request, a real deposit address.

## Funding the account — what this dashboard can and cannot do

**It cannot add funds.** There is no code path anywhere in this project —
dashboard included — that can pull money into the account or push it out.
Depositing is something you do directly on Binance (bank transfer, or
sending crypto from another wallet/exchange). What the dashboard *can*
legitimately do is show you **where** to send it: click "Show deposit
address" to get a real address from your own Binance account via
`get_deposit_address()`. There is deliberately no withdrawal endpoint,
anywhere, matching `execution/exchange_adapter.py`'s own decision not to
implement withdrawal at all — nothing in this system needs to be able to
move funds out of the account it trades with.

Once funds are in the account (via your own deposit), the trading loop uses
whatever balance is actually there — no separate "activate funds" step
beyond your own Binance deposit.

## Security posture

- Binds to `127.0.0.1` by default. `DASHBOARD_HOST=0.0.0.0` is available but
  **do not use it without a reverse proxy providing real authentication and
  TLS** — this Flask dev server has neither.
- Every endpoint that changes state (kill switch) or touches your real
  account beyond a read (deposit address) requires
  `Authorization: Bearer <DASHBOARD_TOKEN>`. Plain status/chart endpoints do
  not, on the assumption that "this machine" is the actual trust boundary —
  if you ever expose this beyond localhost, that assumption breaks and every
  endpoint needs auth, not just the action ones.
- Binance API credentials are loaded and used server-side only; the browser
  never sees them, only JSON results.
- The kill switch button writes/deletes the exact file every `RiskGuardian`
  instance already polls for (`state/KILL_SWITCH`) — verified end-to-end
  against a real `RiskGuardian.check()` call, not just the dashboard's own
  bookkeeping.

## Known limitations

- The price chart's resolution is capped at 1 sample/second by design
  (`price_sample_min_interval_sec` in `PaperExchange.__init__`) to keep the
  table's growth bounded over a long-running deployment — it is not a tick-
  by-tick order book replay.
- Promotion/plasticity-reset events are listed as text next to the RL
  progress chart, not drawn as in-chart vertical lines — Chart.js's
  annotation plugin was deliberately left out to keep this dependency-free
  (one CDN script, not two), and the event list still shows the same
  information.
- No multi-user auth, no HTTPS, no rate limiting on the API endpoints
  themselves (only Binance-side calls are rate-limit-aware). Fine for a
  single operator on their own machine; not fine to expose publicly as-is.
