# Walkthrough: From Zero to a Running System

This is a practical, step-by-step path through TradeJack — every command
below has actually been run during development, not written from a spec. If
a step behaves differently for you, that's worth reporting; the intent is
that this document matches reality.

---

## Step 0 — Install

```bash
git clone <your fork/copy of this repo>
cd TradeJack
pip install -r requirements.txt
```

This installs the CORE dependency set (`torch`, `stable-baselines3`,
`gymnasium`, `ccxt`, `flask`, etc.). GPU-accelerated extras (`cudf`,
`nvidia-dali`) are listed separately at the bottom of `requirements.txt` and
are **not** required — everything degrades gracefully to CPU/NumPy fallbacks
without them (you'll see log lines like `"Laptop Simulation Mode (CPU / Numpy
binary fallback)"`, which is expected and fine on a normal machine).

No GPU, no Docker, and no exchange account are required for anything through
Step 4 below.

## Step 1 — Train and promote your first model

```bash
python -m scripts.train_and_promote --model PPO-DilatedCNN --timesteps 50000
```

What this does:
1. Generates synthetic LOB data under `data_store/` if none exists yet (real
   historical data can be dropped into the same directory structure later —
   see `docs/DATA_FORGE.md`).
2. Trains a real PPO agent via `stable-baselines3` — you'll see standard SB3
   training tables (`ep_rew_mean`, `total_timesteps`, etc.) scroll by.
3. Runs the candidate through `escrow/validation_airgap.py`'s 10-split
   out-of-sample stress test.
4. Promotes to `state/deployed/weights_promoted.zip` **only if** it clears
   the bar (Sharpe ≥ 1.0, drawdown ≤ 15% across all splits).

At 50,000 timesteps on synthetic data, don't expect a promotion — this step
is mainly to confirm the pipeline runs end-to-end on your machine. Two useful
flags for exploring the mechanics without waiting for real promotion:

```bash
# See what a real (not fabricated) "skipped validation" result looks like -- requires --force to actually promote
python -m scripts.train_and_promote --model PPO-DilatedCNN --timesteps 1000 --skip-airgap --force

# Try a different architecture
python -m scripts.train_and_promote --model SAC-DilatedCNN --timesteps 50000
python -m scripts.train_and_promote --model DuelingDQN --timesteps 50000
```

## Step 2 — Run a paper trading session

```bash
python -c "
import asyncio
from execution.live_inference_server import LivePaperInferenceServer

async def main():
    server = LivePaperInferenceServer(
        symbol='BTC-USDT',
        initial_cash=100.0,
        use_synthetic_feed=True,   # set False once you're ready for a real Binance market data connection
        exchange_mode='paper',     # 'paper' | 'testnet' | 'live' -- see Step 5
    )
    await server.run(duration_sec=60.0)

asyncio.run(main())
"
```

This loads whatever's at `state/deployed/weights_promoted.zip` (falling back
to a `MomentumBaseline` with a clear warning if nothing's been promoted yet —
fine for exercising the pipeline, meaningless for judging trading quality)
and runs it against either synthetic data (`use_synthetic_feed=True`, no
network needed) or a real live Binance market-data feed
(`use_synthetic_feed=False` — read-only public data, no API key required for
this part).

Every fill, decision, and price sample lands in
`state/child_900/{fills,decisions}.sqlite` — this is what the dashboard reads.

## Step 3 — Watch it in the dashboard

In a second terminal:

```bash
export DASHBOARD_TOKEN="pick-a-real-secret"   # or leave unset -- one is generated and printed at startup
python -m dashboard.telemetry_server
```

Open `http://127.0.0.1:5000`. You should see the account card update, a price
line with entry/exit markers once trades occur, and (after Step 1) RL
training progress for whichever agents you've trained. See
`docs/DASHBOARD.md` for the full feature list and security notes.

## Step 4 — Run the autonomous continuous loop

Instead of one-shot training (Step 1) and a separate paper session (Step 2),
`genesis_prime.py --mode paper` runs both together: the continuous training
loop (train → evaluate → promote → hot-swap) alongside a live paper-trading
session that automatically picks up each newly-promoted model.

```bash
python -m scripts.genesis_prime --mode paper --capital 100 --symbol BTC-USDT
```

Or, to just run the multi-agent training tournament without a live paper
session attached:

```bash
python -m scripts.genesis_prime --mode crucible --agents 4 --max-steps 50000
```

## Step 5 — Moving toward real money: testnet first, always

**Do not skip this step or shorten it.** Nothing in this codebase has been
tested against a real Binance connection from the environment it was built
in.

1. Create a Binance **testnet** account at https://testnet.binance.vision/
   and generate API credentials there (separate from your real account).
2. Copy `.env.example` to `.env` and fill in your testnet credentials. Prefer
   an Ed25519 key (`BINANCE_ED25519_PRIVATE_KEY_PATH`) over the legacy HMAC
   secret — see `.env.example`'s comments for why and how to generate one.
3. Set `exchange_mode="testnet"` in `scripts/deploy_config.py` (or pass
   `exchange_mode="testnet"` directly to `LivePaperInferenceServer`).
4. Run the same commands as Steps 2 and 4, now against testnet:
   ```bash
   python -m scripts.genesis_prime --mode testnet --capital 100
   ```
5. Watch it run for **at least `min_weeks_testnet_before_live` weeks**
   (2, by default — `scripts/deploy_config.py`) before even considering
   `exchange_mode="live"`. Use `execution/session_report.py` and the
   dashboard to judge whether it's actually clearing the bar (beats
   fee-adjusted buy-and-hold with statistical significance), not just
   whether it ran without crashing.
6. Only after that: real credentials, `exchange_mode="live"`, and start with
   an amount you are fully prepared to lose. The kill switch
   (dashboard button, or `touch state/KILL_SWITCH`) halts all trading
   immediately at any point.

## Step 6 (optional) — Two sleeves: scalper + position

```python
import asyncio
from execution.multi_sleeve_orchestrator import MultiSleeveOrchestrator
from execution.sleeve_config import DEFAULT_HFT_SLEEVE, DEFAULT_POSITION_SLEEVE

async def main():
    orch = MultiSleeveOrchestrator(
        symbol="BTC-USDT",
        sleeve_configs=[DEFAULT_HFT_SLEEVE, DEFAULT_POSITION_SLEEVE],
        total_capital=1000.0,
        use_synthetic_feed=True,  # False for real data
    )
    await orch.run(duration_sec=120.0)

asyncio.run(main())
```

Read `execution/sleeve_config.py`'s comments before using this for real —
the default tick-count assumptions there are calibrated for Binance's public
100ms depth stream and should be checked against your actual measured update
rate, not assumed.

## Troubleshooting

- **`No promoted weights found... Running fallback MomentumBaseline`**: expected
  if you haven't completed Step 1 with a checkpoint that actually cleared the
  validation bar. Not an error.
- **`pandera not installed. Schema validation will be skipped`**: expected and
  fine — an optional dependency, degrades gracefully.
- **Training seems slow**: SB3 training speed depends heavily on hardware and
  batch size; the defaults are tuned for correctness/testing, not throughput.
  See `swarm/model_registry.py`'s `default_hyperparams` per card to tune.
- **Dashboard shows no data**: confirm `TRADEJACK_STATE_DIR` (dashboard) points
  at the same `state_dir` your training/trading session actually used.
- **Something behaves differently than documented here**: check
  `docs/project_history.md` first — it's the ground truth for what's been
  verified vs. what's still aspirational, in far more detail than any single
  doc page can hold.
