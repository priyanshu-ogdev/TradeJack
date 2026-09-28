# TradeJack

**TradeJack** is a continuously self-training reinforcement-learning trading
system: it trains PPO/SAC/DQN agents on limit-order-book data, evaluates and
promotes checkpoints through a statistical validation gate, and executes the
promoted policy's buy/sell/hold decisions against either a realistic paper
wallet or a real Binance Spot account — with a live telemetry dashboard to
watch it happen.

This README describes what the system **actually does today**, verified by
running it, not what it was originally envisioned to become. For the full,
honest history of what was found broken and fixed along the way, see
[`docs/project_history.md`](docs/project_history.md) — it is deliberately
unflattering where it needs to be. For a step-by-step guide to actually
running this, see [`docs/WALKTHROUGH.md`](docs/WALKTHROUGH.md).

---

## What it does

1. **Trains** three real RL algorithms (PPO, SAC, Dueling DQN) via
   `stable-baselines3` on a limit-order-book simulation (`physics/lob_env.py`),
   sharing a causal dilated-CNN feature encoder. Alongside the RL agents, it
   now trains a **Supervised Predictor** (`training/supervised_predictor.py`)
   to act as a reality-check against RL hallucinations.
2. **Protects against catastrophic forgetting** with Elastic Weight
   Consolidation computed correctly per-algorithm (log-likelihood Fisher for
   PPO and SAC's actor, TD-loss Fisher for DQN and SAC's critic — not a
   one-size-fits-all approximation), and **counters primacy bias / loss of
   plasticity** with periodic, Fisher-guided partial resets of policy heads
   during continuous training.
3. **Validates before promoting**: a walk-forward evaluator runs a Mann-Whitney
   U significance test against buy-and-hold and momentum baselines before any
   checkpoint reaches deployment; the gate fails *closed* if it can't run
   (missing dependency, skipped flag) rather than defaulting to "pass."
4. **Composes Signals and Orchestrates Portfolios**: The `LiveComposer`
   dynamically fuses the RL policy's intent with the supervised predictor's
   confidence and risk limits (using independent greedy/cautious axes). The
   `PortfolioOrchestrator` distributes capital dynamically across multiple
   instruments based on these composed signals and volatility tracking.
5. **Executes real decisions** against either a paper wallet with realistic
   order-book-depth fills, latency, and fees (`execution/paper_exchange.py`)
   or a real Binance Spot account (`execution/exchange_adapter.py` +
   `execution/live_exchange_bridge.py`) — controlled by one config field
   (`exchange_mode: paper | testnet | live`), with the same risk-guardian
   safety layer (kill switch, position limits, daily-loss halt, drawdown
   halt) in front of both.
6. **Shows you what's happening and lets you control it** in a live **React Control Panel**
   (`frontend/control-panel` + `dashboard/telemetry_server.py`): price with real entry/exit markers,
   equity curve, per-agent RL training progress, recent risk decisions, and
   your real Binance balance. You can start/stop trading and training via a
   decoupled, file-based command channel that never blocks the trading process.

## What it does not do, on purpose

- **It cannot deposit or withdraw funds.** Nothing in this codebase — the
  dashboard included — can move money into or out of your Binance account.
  You fund the account yourself, directly on Binance; the dashboard can only
  show you your balance and where to send a deposit.
- **It cannot short.** This trades spot only. Every layer (training
  environment, paper wallet, live execution bridge) clamps any target
  position that would require a short to "fully exit," never negative.
- **It has never placed a real order.** Every real-exchange code path in this
  repository has been built and unit-tested against mocks; none of it has
  been run against an actual Binance connection from the environment this was
  developed in (no outbound network access to Binance's domains was
  available). **Test on Binance testnet, extensively, before real capital.**
  `scripts/deploy_config.py`'s `min_weeks_testnet_before_live` exists for
  exactly this reason.
- **It does not guarantee profit.** The actual bar, stated the same way since
  the very first design discussion: a promoted checkpoint should beat
  fee-adjusted buy-and-hold with statistical significance over a real
  multi-week walk-forward window — that is a research/engineering standard,
  not an outcome this document promises.

---

## Repository structure

```
TradeJack/
├── physics/                    # The training environment
│   ├── lob_env.py              #   TradeJackLOBEnv: gym env, reward/penalty design, long-only clamp
│   └── portfolio_tracker.py    #   PortfolioAccountingEngine: SQLite ledger, Sharpe/Sortino/drawdown
├── swarm/                      # The RL training engine
│   ├── model_registry.py       #   6 model cards: PPO-Transformer, PPO-DilatedCNN, SAC-DilatedCNN,
│   │                           #   DuelingDQN, Momentum-Baseline, BuyAndHold-Baseline
│   ├── shared_encoder.py       #   LOBFeatureExtractor: causal dilated-CNN, shared across all SB3 policies
│   ├── rl_trainer.py           #   OnlineRLTrainer: thin, correct wrapper around SB3's .learn()/.save()/.load()
│   ├── ewc_sb3_adapter.py      #   PolicyEWC: per-algorithm Fisher computation (PPO/SAC/DQN)
│   ├── sb3_replay_buffer_adapter.py  # Wires prioritized+HER replay into SAC/DQN's actual training
│   ├── plasticity_manager.py   #   Periodic Fisher-guided head resets against primacy bias
│   ├── training_progress_ledger.py  # Durable SQLite record of training progress, for the dashboard
│   └── ...                     #   self_mod_manager, git_rollback, social_relay: earlier-generation
│                                #   components, largely superseded by the SB3 pipeline above — see
│                                #   docs/project_history.md for what's still live vs legacy
├── training/                   # Orchestrates the RL training engine
│   ├── crucible_tournament.py  #   Multi-agent PBT tournament across the model roster
│   ├── continuous_trainer.py   #   Background loop: train -> evaluate -> promote -> hot-swap live model
│   ├── supervised_predictor.py #   Supervised model for signal composition
│   └── walk_forward_evaluator.py  # Statistical promotion gate (Mann-Whitney U vs baselines)
├── execution/                  # Real and paper trade execution
│   ├── paper_exchange.py       #   Realistic paper wallet: book-depth fills, latency, real fee schedule
│   ├── exchange_adapter.py     #   BinanceSpotAdapter: real orders via ccxt, rate-limit aware, Ed25519
│   ├── live_exchange_bridge.py #   Bridges BinanceSpotAdapter into the same interface the trading loop uses
│   ├── live_inference_server.py#   The actual decision loop: feed -> model -> risk check -> exchange
│   ├── composition_layer.py    #   LiveComposer: dynamic signal fusion (RL + Predictor + Risk)
│   ├── portfolio_orchestrator.py # Distributes capital across instruments
│   ├── command_channel.py      #   Decoupled file-based command channel for the dashboard
│   ├── risk_guardian.py        #   Kill switch, position/rate limits, daily-loss and drawdown halts
│   ├── binance_live_feed.py    #   Real Binance depth+trade WebSocket feed
│   ├── shared_feed_hub.py      #   One live connection fanned out to multiple sleeves
│   ├── multi_sleeve_orchestrator.py  # Runs an HFT-scalper + position-swing sleeve concurrently
│   └── session_report.py       #   P&L / Sharpe / significance report for a session
├── frontend/control-panel/     #   React frontend for monitoring and control
├── dashboard/
│   └── telemetry_server.py     #   Flask API backing the React dashboard, uses command channel
├── escrow/                     # Validation gate (used by the manual promotion script)
│   └── validation_airgap.py    #   Multi-split out-of-sample stress test before promotion
├── data_forge/                 # Historical/synthetic data ingestion (GPU-accelerated where available)
├── warden/                     # Legacy hypervisor components (VRAM tiering, OOM watchdog) — see
│                                # docs/project_history.md for current relevance
├── scripts/
│   ├── genesis_prime.py        #   Main orchestrator: --mode crucible | paper | testnet | live
│   ├── process_supervisor.py   #   Supervisor for live/testnet (handles SIGTERM and restarts)
│   ├── train_and_promote.py    #   One-shot manual: train -> validate -> promote CLI
│   └── deploy_config.py        #   DeploymentConfig: exchange_mode, risk limits, testnet gating
├── tests/                      # Unit tests (does not yet cover the 3 newest swarm/ modules — see
│                                # docs/DEPLOYMENT_AND_TESTING.md)
└── docs/                       # See below
```

## Documentation

- **[docs/WALKTHROUGH.md](docs/WALKTHROUGH.md)** — start here: install, generate data, train, validate,
  run a paper session, view the dashboard, move to testnet.
- **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** — full system design: data flow, the RL training
  engine, the execution layer, the dual-sleeve design, and what "high-frequency" honestly means here.
- **[docs/DASHBOARD.md](docs/DASHBOARD.md)** — dashboard setup, security posture, and what it can/can't do.
- **[docs/DEPLOYMENT_AND_TESTING.md](docs/DEPLOYMENT_AND_TESTING.md)** — running the test suite,
  known coverage gaps.
- **[docs/SWARM_EVOLUTION.md](docs/SWARM_EVOLUTION.md)** — deep dive on EWC, plasticity management,
  and the replay buffer.
- **[docs/ESCROW_AND_AIRGAP.md](docs/ESCROW_AND_AIRGAP.md)** — the validation/promotion gate in detail.
- **[docs/DATA_FORGE.md](docs/DATA_FORGE.md)** — data ingestion pipeline.
- **[docs/WARDEN_HYPERVISOR.md](docs/WARDEN_HYPERVISOR.md)** — the legacy VRAM-tiering/watchdog subsystem.
- **[docs/FULL_DESIGN_REVIEW.md](docs/FULL_DESIGN_REVIEW.md)** — details the architecture correction separating the dashboard control plane from the trading loop.
- **[docs/COMPOSITION_LAYER.md](docs/COMPOSITION_LAYER.md)** — deep dive into dynamic signal sizing and independent greed/caution axes.
- **[docs/PROCESS_SUPERVISION.md](docs/PROCESS_SUPERVISION.md)** — how the trading process handles graceful shutdown and restarts.
- **[docs/project_history.md](docs/project_history.md)** — the complete, honest changelog: every bug
  found, how it was found, and how it was verified fixed. Read this if you want to know exactly how
  much to trust any given part of the system.

## Quick start

```bash
pip install -r requirements.txt
python -m scripts.train_and_promote --model PPO-DilatedCNN --timesteps 50000
python -m dashboard.telemetry_server   # in a second terminal
```

Open `http://127.0.0.1:5000`. See [docs/WALKTHROUGH.md](docs/WALKTHROUGH.md) for the full path
including autonomous continuous training and moving to testnet.

## Current defaults worth knowing before you change anything

| Setting | Default | Where |
|---|---|---|
| Starting capital | $100 | `scripts/deploy_config.py` |
| Max position size | 50% of equity | `scripts/deploy_config.py` |
| Minimum hold between flips | 100 ticks | `scripts/deploy_config.py` |
| Exchange mode | `paper` | `scripts/deploy_config.py` (`paper`/`testnet`/`live`) |
| Weeks on testnet required before live | 2 | `scripts/deploy_config.py` |
| Promotion Sharpe / drawdown bar | ≥1.0 / ≤15% across 10 splits | `escrow/validation_airgap.py` |
| Taker fee assumption | 0.10% (Binance VIP0 default) | `physics/lob_env.py`, `execution/paper_exchange.py` |

## License

Proprietary — see `LICENSE`.
