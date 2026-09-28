# Project TradeJack: Comprehensive Development History & Roadmap

This document serves as the master chronological log of all architectural decisions, implementations, and verified milestones for **Project TradeJack**.

---

## 🏛️ 1. Project Genesis & Core Constraints
**The Vision:** Build a completely sovereign, self-contained AI trading swarm that evolves dynamically without human intervention.
**The Inspirations:** Fusing the autonomous survival, self-modification, and P2P social relay mechanics from `Conway-Research/automaton` with the advanced predictive neural architectures from `huseinzol05/Stock-Prediction-Models`.
**The Hardware Target:** Uncompromisingly designed for a local bare-metal **NVIDIA DGX Spark** (1 PetaFLOP, 128GB Unified Memory, Grace Blackwell, CUDA 13). No cloud costs, no MT5 demo accounts—just 100% hardware saturation 24/7.
**The Survival Physics:** Each swarm agent (Child) spawns with exactly $10.00. They face a relentless logarithmic stagnation tax ($- \alpha \ln(1+t) - \beta t_{\text{stag}}$) and must trade profitably across a synthetic Limit Order Book (LOB) to survive.

---

## 🏗️ 2. Phase 1: Environment & Physics Engine (Completed)
We first built the underlying reality that the swarm agents live in.
- **KvikIODataForge (`data_forge/`)**: Built a high-throughput, zero-copy synthetic LOB data pipeline capable of generating realistically noisy order book depth states (150+ ticks/day). Fully fallback-compatible with Numpy/CPU when CUDA is unavailable during laptop development.
- **TradeJackLOBEnv (`physics/lob_env.py`)**: A vectorized, high-performance Gymnasium-style environment mirroring real-world slippage, latency, and maker/taker fees. 
- **Portfolio Accounting (`physics/portfolio_tracker.py`)**: Strict immutable ledger to track live equity, max drawdown, Sharpe, and Sortino ratios for every container.

---

## 🧠 3. Phase 2: Hardware Hypervisor & Warden (Completed)
To prevent the 50 Docker containers from crashing the GPU via Out-Of-Memory (OOM) errors, we built the ultimate hardware gatekeeper.
- **Warden Core (`warden/warden_core.py`)**: Implemented NVIDIA MIG (Multi-Instance GPU) partitioning logic to enforce strict VRAM memory tiers across containers.
- **Memory Tiers**: 
  - **Tier 1 (High Alpha)**: 20GB VRAM allowed (for complex sequence models).
  - **Tier 2 (Standard Compute)**: 4GB VRAM allowed.
  - **Tier 3 (Survival Mode)**: 1GB / Inference-only / FP8 allowed.
- **OOM Watchdogs & Penalties**: Hardware watchdogs that instantly detect CUDA OOM, enforce a $10 financial penalty, and forcefully drop the violating container down to Tier 3.

---

## 🔐 4. Phase 3: P2P Swarm Network & Trustless Escrow (Completed)
We implemented the social relay and security mechanisms for autonomous knowledge sharing.
- **Social Relay Bridge (`swarm/social_relay.py`)**: Built a P2P gossip protocol where containers can advertise their successful weights (lineages) using cryptographically secured HMAC-SHA256 signatures, random nonces, and ULID packet IDs to prevent replay attacks.
- **Trustless Escrow (`escrow/escrow_contract.py`)**: When a struggling agent wants to buy weights from a successful peer, funds (e.g., \$2.00 USDC) are temporarily locked in `escrow_ledger.sqlite`.
- **10x Validation Airgap (`escrow/validation_airgap.py`)**: Before funds settle and weights are transferred, the Warden sandboxes the weights and forces them through 10 out-of-sample stress data splits. If the model is poisoned or overfitted (Sharpe < 1.0 or Drawdown > 15%), the transaction is instantly rejected and the buyer is 100% refunded.

---

## 🧬 5. Phase 4: Universal Model Registry & Neuro-Evolution (Completed)
We fused the actual neural architectures into the swarm container logic.
- **Universal Model Registry (`swarm/model_registry.py`)**: Implemented 21+ neural architectures (Transformers, CNNs, VAEs, Actor-Critic, Q-Learning, Neuro-Evolution, Stacking Ensembles) categorized strictly into the 3 hardware tiers.
- **Dynamic Self-Modification (`swarm/self_mod_manager.py`)**: Containers dynamically detect their Warden VRAM tier and equity state (`HIGH`, `NORMAL`, `LOW_COMPUTE`, `CRITICAL`). If an agent bleeds cash and hits `CRITICAL`, the `SelfModEngine` automatically downgrades its neural architecture (e.g., from a heavy Transformer to a lightweight `Curiosity-Q-learning-Agent`) to minimize compute tax and survive.
- **PBT Architecture Mutation (`swarm/rl_mechanics.py`)**: During Population-Based Training, failing agents do not just copy weights; they actively dump their entire neural architecture (`model_name`) to inherit the successful parent's architecture, driving true macro-evolution across the swarm.
- **Git Rollbacks**: Automatic HWM (High-Water Mark) checkpoints and instant codebase rollbacks if a container suffers a drawdown >15%.

---

## 🚀 6. Phase 5: Genesis Orchestration & SOTA Hardening (Completed)
We successfully integrated the Data Forge, Physics Engine, Warden, and Swarm into the master `genesis_prime.py` orchestrator and sealed the "5 Silent Killers" of bare-metal execution:
- **The Reaper Protocol**: OS-level signal handlers (`SIGINT`, `SIGTERM`, `atexit`) intercept aborts and violently tear down the vLLM Shared Brain and Warden processes, preventing Zombie containers from holding VRAM hostage.
- **Strict Environment Masking**: The master process blanks `CUDA_VISIBLE_DEVICES` after hardware validation, physically restricting PyTorch context fragmentation on the DGX Grace Blackwell Unified Memory pool.
- **Aggressive CUDA Purging**: During `SelfModEngine` model swaps, the agent actively pushes weights to CPU, executes `gc.collect()`, deletes the python `sys.modules` reference, and sweeps `torch.cuda.empty_cache()` to permanently eliminate memory creep.
- **Multiprocessing SQLite Audit Isolation**: The Warden's tax assessment loop (`run_audit_cycle`) was pulled out of Python's Thread GIL and injected into an isolated `multiprocessing.Process`. This guarantees zero micro-stutters on the FastAPI network when executing heavy SQL joins across 50 agent databases simultaneously.
- **Idempotent Taxation**: SQLite schemas in `portfolio_tracker.py` and `warden_core.py` now enforce `UNIQUE(child_id, market_timestamp, tax_type)` with `INSERT OR IGNORE` logic, making asynchronous double-dip taxation mathematically impossible.

---

## 🔥 7. v3 Upgrade: The Honest Deployment Integration (Completed)
The ultimate transition from theoretical simulation to a production-grade, gradient-backed live environment. We adopted the core principle: *"The Crucible evolves. The Deployment freezes. The evidence decides."*

- **Standardized RL Engine (`stable-baselines3`)**: Eradicated all 24 legacy toy architectures and custom PyTorch forward loops. Implemented `PPO-DilatedCNN`, `SAC-DilatedCNN`, and `DuelingDQN` natively wrapping `stable-baselines3`. Integrated `PrioritizedReplayBuffer` for off-policy agents.
- **Shared Backbones**: Built `LOBFeatureEncoder` to collapse sequence variables into a unified latent space for all policy heads.
- **Gymnasium Physics Upgrade**: Rewrote `lob_env.py` (TradeJackLOBEnv) to perfectly align with strict Gymnasium APIs, exposing standard `observation_space` and `action_space` boundaries to the SB3 actors.
- **Live Execution Pipeline**: 
  - Created `ExchangeAdapter` interfaces with a strict `PaperExchangeAdapter` fallback.
  - Implemented the **`RiskGuardian`**: An immutable safety layer that blocks illegal orders, respects max drawdowns, checks daily loss caps, and prevents rate-limit explosions.
  - Built the **`PositionThrottle`**: Modulates trading sizes linearly based on a rolling 24h Sortino ratio.
- **Self-RL Continuous Loop**: Built `ContinuousTrainer` to train on inference telemetry alongside `CrucibleTournament` to manage PBT exploiting and mutation for top performers.
- **Verified Promotion Gates**: Checkpoints generated by SB3 are subjected to the `ValidationAirgapEngine`, which utilizes rigorous statistical Mann-Whitney U test comparisons over buy-and-hold benchmarks before unlocking live testnet deployment logic in `genesis_prime.py`.

**Final Status**: Systems Nominal. Project TradeJack v3 is sealed, verified, and fundamentally immortal on bare-metal execution. Ready for Phase 4 (Observability).

> **Correction, added after actually running this pipeline rather than reading it**: several items above were description of intent rather than verified behavior at the time they were written. Specifically: the "Verified Promotion Gates" claim shipped with a validation engine whose stress splits were bit-for-bit identical regardless of `num_splits` (a seed that was set but never consumed by the data-selection logic); `PrioritizedReplayBuffer`/HER was built but never actually wired into SAC/DQN's training calls; `ewc_optimizer.py`'s EWC was never constructed anywhere in the SB3 pipeline, so it provided zero protection against forgetting despite `EWCCallback` existing; and the manual training script's `--skip-airgap` flag defaulted to a fabricated passing result that could trigger promotion without `--force`. All four are fixed and re-verified as of Section 8 below — noting this here rather than quietly rewriting history, since the gap between "the class exists" and "the class is actually being used correctly by the running system" is exactly the kind of thing worth being honest about in a changelog.

---

## 🔬 8. v3.1 Upgrade: Closing the Gap Between Designed and Actually-Running (Completed)

Everything in this phase was found by *running* the pipeline end-to-end and inspecting real output, not by reading class names or docstrings — several of the following bugs were completely invisible from a code review alone and only surfaced from actual execution traces.

**Training engine, now genuinely closed-loop:**
- Fixed the validation airgap's stress splits actually varying per split (seeded random warm-up skip before each split's measured window) — previously confirmed bit-for-bit identical across all splits.
- Fixed `train_and_promote.py --skip-airgap` fabricating a passing result that could silently promote without `--force`.
- Wired `swarm/sb3_replay_buffer_adapter.py`'s `SB3ReplayBufferAdapter` so SAC/DQN actually train against the prioritized+HER buffer instead of SB3's default uniform buffer.
- Extended EWC to actually work for SB3 policies (`swarm/ewc_sb3_adapter.py`'s `PolicyEWC`) across all three algorithms — PPO (log-likelihood Fisher), DQN (TD-loss Fisher, matching Kirkpatrick et al.'s original DQN-Atari treatment), and SAC (both actor log-likelihood and critic TD-loss, matching Powers et al. 2021). Fixed a second bug found while testing this: the EWC correction step assumed a single `policy.optimizer`, silently no-opping every SAC cycle (SAC has separate actor/critic optimizers).
- Fixed `OnlineRLTrainer.save()` failing with `cannot pickle 'generator' object` on **every single checkpoint save, for every architecture** — root-caused to a custom `_tradejack_callbacks` attribute bolted onto the SB3 model object, which SB3's generic pickle-everything save tried to serialize along with its own state.
- Added `swarm/plasticity_manager.py`'s `PlasticityManager`: periodic, Fisher-guided partial resets of policy/value/Q head layers (never the shared encoder) to counter primacy bias / loss of plasticity during continuous training on non-stationary market data (Nikishin et al. 2022, extended with Fisher-guided selectivity per arxiv 2502.00802).

**Live execution layer, now actually connected:**
- Added `LivePaperInferenceServer.hot_swap_model()` — `ContinuousTrainer._promote_champion()` was already calling this via a `hasattr()` guard, but the method didn't exist, so every promotion silently no-opped on the live side.
- Fixed `genesis_prime.py --mode paper/testnet/live` crashing immediately (`AttributeError: 'LivePaperInferenceServer' object has no attribute 'run_forever'`) — the orchestrator and the server had been written against incompatible interfaces; this code path had never been successfully run before.
- Fixed a misleading docstring on `BinanceSpotAdapter.place_market_order` claiming it raises on exchange errors when it actually returns a `status="rejected"` result — verified every current caller already checks `.status` correctly, but a future caller trusting the docstring wouldn't have.
- Replaced ~39 hardcoded Windows-style `"d:/TradeJack/..."` path defaults across 16+ files with portable relative paths — confirmed these silently produced nonsense nested paths on Linux (`<cwd>/d:/TradeJack/state/...`) rather than failing loudly, since Linux has no drive-letter concept and treats the string as a relative path segment.

**Dependency manifest**: `requirements.txt` was missing `ccxt` entirely despite it being load-bearing for `BinanceSpotAdapter`'s real order placement. Rebuilt the manifest with a clear CORE / optional-GPU split, since several DGX-tier dependencies (`cudf`, `nvidia.dali`, `chromadb`, `pandera`) are not reliably plain-pip-installable and already degrade gracefully in code via try/except.

**Status, stated precisely rather than declared "immortal" again**: the training engine now produces real, saveable, loadable, promotable checkpoints across all three algorithms, verified by actually running full tournament cycles end-to-end. The live-paper execution path runs without crashing. What is *not* yet true: `PolicyEWC`'s Fisher-guided plasticity reset interval is a literature-informed default, not tuned against this project's real data; there is no formal `tests/` coverage for the three new `swarm/` modules (self-tests only); and no real-money order has ever been placed or tested against a live exchange connection from this environment (network-restricted; must be validated on a machine with real connectivity before trusting `BinanceSpotAdapter` end-to-end).

---

## 🔬 9. v3.2 Upgrade: Binance Live-Trading Research & Exchange Adapter Hardening (Completed)

Researched Binance's *current* API documentation directly rather than relying on training-data knowledge that can't be trusted to be current for a fast-moving exchange integration — and found one change significant enough to matter for correctness:

- **Binance discontinued the REST `listenKey` mechanism for Spot user data streams entirely as of 2026-02-20 07:00 UTC** (`POST/PUT/DELETE /api/v3/userDataStream` no longer works). The replacement is subscribing to the user data stream directly through the WebSocket API (`userDataStream.subscribe`, a signed request), not a REST listenKey you poll/keepalive. Confirmed this codebase never implemented the old mechanism in the first place (so nothing here was broken by the deprecation), but documented the correct *current* approach prominently in `execution/exchange_adapter.py`'s module docstring specifically so nobody later copies example code for the now-nonexistent old pattern.
- **Deliberately did not hand-roll the new signed WebSocket API user data stream** in this pass — implementing a signed WS API session correctly, with no way to test it against a real Binance connection from this network-restricted environment, is exactly the kind of real-money-adjacent code worth building carefully with real connectivity to verify against, not rushing. Stated as an open gap in the adapter's class docstring rather than silently left unmentioned.
- **Added `BinanceRateLimitTracker`**, per Binance's own explicit guidance ("Please use WebSocket Streams for live updates to avoid bans") and documented rate-limit values (6,000 weight/minute per IP): reads `X-MBX-USED-WEIGHT-*` from actual response headers (supplementing ccxt's own static-table throttling with what the server actually reports), and — the more safety-critical half — **distinguishes HTTP 418 (IP already banned, 2 minutes to 3 days depending on repeat offenses) from HTTP 429 (back off) from an ordinary order rejection**. The previous version caught all three identically as "order failed, status=rejected," meaning a banned IP would have kept getting hit with the next signal, extending the ban. `place_market_order` now raises `ExchangeBannedError` proactively before even attempting a call while inside a known ban window.
- **Added pre-flight exchange filter validation** (`get_symbol_filters`/`_validate_order_against_filters`) using ccxt's real, loaded-at-`connect()`-time market data (`limits.amount.min`, `limits.cost.min`) rather than the illustrative placeholder constants `paper_exchange.py` uses for local logic-testing — an order that would be rejected by Binance's own `LOT_SIZE`/`MIN_NOTIONAL` filters is now caught locally before spending a request and rate-limit weight on it.
- Verified the fee-schedule assumption already in `paper_exchange.py` (0.1%/0.1% VIP0 maker/taker) is still Binance's current default as of this research pass — no change needed there.
- Fixed an inconsistency introduced partway through this same upgrade: `get_open_orders`/`cancel_all_orders` weren't feeding the new rate-limit tracker like every other method was updated to. Also made `RiskGuardian.execute_safe_order` catch `ExchangeBannedError` and return a proper `OrderResult(status="banned")` instead of letting the exception propagate uncaught — verified with a mock banned exchange.

---

## 🔬 10. v3.3 Upgrade: Broader Live-API Survey + Ed25519 Migration (Completed)

Surveyed the current (2026) landscape of free-to-use trading APIs suitable for Python bots — Bybit, Kraken, Coinbase Advanced, KuCoin, OKX, and RoboForex — to confirm Binance remains the right choice rather than assuming it by inertia:

- **RoboForex** is a forex/CFD broker, not a crypto spot exchange — its API surface is primarily MetaTrader 4/5 (MQL-based) plus a proprietary "R StocksTrader" API, with no first-class Python SDK comparable to `ccxt`. Not a fit for this project's BTC-USDT spot focus.
- **OKX has a genuinely notable demo-trading API** worth knowing about even though not adopted here: the *same* live endpoints work in a fully simulated mode via a single `x-simulated-trading: 1` header, using real live market data — a cleaner "paper trading via the real API surface" story than most alternatives. Recorded as a real, verified alternative worth revisiting if Binance-specific constraints ever become limiting.
- **Binance remains the right choice for this project**: free (no-cost API access, just an account), fastest public update cadence among the surveyed venues for this use case (100ms depth stream), and `ccxt` support is mature. No change of exchange recommended.

**A real, actionable finding from that research**: Binance's own current documentation states plainly that "**HMAC keys are deprecated. We recommend to migrate to asymmetric API keys, such as Ed25519**" — smaller and faster to verify than RSA, and unlike HMAC, no shared secret is ever transmitted at all (asymmetric — only the public key goes to Binance). Checked `ccxt`'s actual binance signing code (not assumed): it already auto-detects a PEM-formatted key in the `secret` field and switches to Ed25519/RSA signing automatically — no ccxt changes were needed, only how this adapter loads credentials.

`BinanceSpotAdapter` now supports loading an Ed25519 private key file (`BINANCE_ED25519_PRIVATE_KEY_PATH`), preferred over the legacy HMAC secret (`BINANCE_API_SECRET`, still supported but now logs a deprecation warning when it's the only credential present). Verified end-to-end with a real, locally-generated Ed25519 keypair: correct PEM detection, correct routing through ccxt's own length-based Ed25519-vs-RSA branch, and correct fallback to HMAC when no asymmetric key is configured. `.env.example` updated to document both paths with the asymmetric one presented first.

---

## 🔬 11. v3.4 Upgrade: Telemetry Dashboard + Dual-Sleeve Re-Integration (Completed)

**Re-merged a dropped feature, found during this pass**: `execution/sleeve_config.py`, `shared_feed_hub.py`, and `multi_sleeve_orchestrator.py` (the dual HFT-scalper + position-swing architecture, designed and verified two upgrade rounds ago) had been silently lost during an earlier file-consolidation pass — they existed in one working checkout that was never fully merged in. Found this specifically because `execution/live_inference_server.py` had since evolved (the `hot_swap_model` fix, threading lock) in a *different* checkout that never had the sleeve-support parameters (`owns_feed`, `decision_interval_ticks`, `risk_limits_override`) added back. Merged both sets of changes together carefully rather than picking one file version over the other, and re-verified the full multi-sleeve flow end-to-end (both sleeves running concurrently against one shared feed, correct capital allocation, correct decision cadence, correct filter-mismatch handling) against the current, more-evolved codebase.

**`swarm/training_progress_ledger.py`**: a new, durable, queryable SQLite store for RL training progress (Sortino, equity, cumulative timesteps per cycle, per agent) and events (promotions, plasticity resets) — closing the gap that `TrainingMetricsCallback`'s in-memory-only history couldn't survive a process restart or be read from outside the training process. Wired into `training/crucible_tournament.py` (per-cycle) and `training/continuous_trainer.py` (promotion and plasticity-reset events). Verified real training cycles actually land in the ledger, not just compile.

**`execution/paper_exchange.py`** gained a lightweight, separate `price_samples` table (throttled to at most 1 sample/second, deliberately not touching `PortfolioAccountingEngine`'s own ledger schema that other consumers depend on) — needed so a dashboard can plot a real price line under entry/exit trade markers instead of only isolated, disconnected trade points.

**`dashboard/telemetry_server.py` + `dashboard/templates/index.html`**: a Flask dashboard, run as a fully separate process from the trading loop (reads the same SQLite files the trading loop writes — no in-process coupling, no asyncio-vs-Flask threading complexity). Shows: live account cards, a price chart with real entry/exit markers, the equity curve, per-agent RL training progress (Sortino/equity per cycle with promotion/reset events listed), a recent risk-decisions log, and a Binance balance/deposit-address section. Verified every endpoint against real generated data (a real paper-trading session and real training cycles), not just that the code compiles — including confirming the dashboard's kill-switch button actually halts a real, separately-constructed `RiskGuardian` instance via the same file both already agree on.

**On "adding funds," addressed directly and precisely**: confirmed there is no way for this system to pull funds into the account or push them out — funding is always a manual action on Binance itself. The dashboard's Binance section reflects this honestly: it shows current balance and, on request (token-gated, since it's still a real account-touching call), a real deposit address via `get_deposit_address()` — never a fabricated "add funds" action. No withdrawal endpoint exists anywhere in the dashboard, matching `execution/exchange_adapter.py`'s own stance.

---

## 🔬 12. CRITICAL FIX: The RL Model Was Never Actually Connected to a Real Exchange (v3.5)

Directly asked to verify "is the whole workflow seamlessly connected for real trading" — traced it end to end rather than assuming, and found the most significant gap of this entire upgrade series.

**The finding**: `LivePaperInferenceServer.__init__` unconditionally constructed `PaperExchange`, regardless of `DeploymentConfig.exchange_mode` ("paper"/"testnet"/"live"). Confirmed by grepping every construction site of `BinanceSpotAdapter` across the entire repository: it was only ever instantiated in `dashboard/telemetry_server.py`'s read-only balance display and in the adapter's own self-tests. **Zero call sites anywhere connected it to the actual trading decision loop.** Despite extensive, careful work across earlier rounds — Ed25519 key support, rate-limit tracking, exchange-filter validation, deposit-address lookup — the RL model's buy/sell/hold decisions had no path to a real exchange under *any* configuration, including `exchange_mode="live"` with real API keys configured. It would have silently kept paper trading forever.

**The fix, `execution/live_exchange_bridge.py`**: a bridge class presenting `PaperExchange`'s exact interface (`submit_target_position`, `.accounting.equity`, `.position_qty`, `on_depth_update`, `book_is_stale`) while placing real orders through `BinanceSpotAdapter` underneath — chosen over rewriting the trading loop itself specifically to avoid any risk of changing behavior on the already-extensively-tested paper path. Handles the real sync/async mismatch this connection requires: `PaperExchange` can answer "what's my equity" instantly from local state, a real exchange cannot without a network round-trip — resolved with a locally-cached equity/position estimate, updated immediately from every real fill's own response, corrected periodically (every 30s) by a background reconciliation against actual account balance. `LivePaperInferenceServer` now reads `exchange_mode` and constructs the right exchange object; nothing else about the trading loop changed.

**A second, independent bug found while building and testing the bridge, in `PaperExchange` itself**: `target_frac`'s `[-1, 1]` range implicitly assumes short-selling is possible. This is a **spot** exchange — there is no shorting, a position can never go negative in reality. `PaperExchange.submit_target_position` had no such constraint and was silently allowing the model to "sell" more than it held, simulating short positions that could never exist on a real account. This means **every past paper-trading result involving strongly negative target values was more optimistic than reality could ever support** — caught only because the real exchange adapter naturally rejects an oversized sell (insufficient balance) where the local simulation didn't. Fixed with a long-only clamp in both `PaperExchange` and `LiveExchangeBridge`: any target implying a short is now capped at fully exiting the position, never further.

**A logic bug in the fix itself, caught immediately by its own test**: the first version of the `exchange_mode` branch also checked `use_synthetic_feed`, conflating two orthogonal concerns — which market *data* feed to use, and which *exchange* to send orders to. This incorrectly forced paper mode whenever testing with a synthetic feed, making it impossible to test the new live-exchange wiring without a real network connection. Fixed to depend only on `exchange_mode`.

**Verified, precisely**: paper mode regression-tested unchanged (real synthetic session, zero errors). The new `testnet`/`live` path tested end-to-end against a mock `BinanceSpotAdapter` (this sandbox has no route to Binance's real domains) — confirmed `LiveExchangeBridge` is actually constructed, real balance reconciliation fires and reports correctly, a full inference session runs without error, and the long-only clamp correctly prevents a short. **Not yet tested against a real Binance connection** — this is exactly the kind of change that needs real testnet validation, extensively, before anything resembling real capital, matching this project's own `min_weeks_testnet_before_live` philosophy. If you do nothing else before going live, run this specific path on testnet first.

---

## 🔬 13. RL Layer Review: Two More Train/Serve Mismatches Found at the Source (v3.6)

Asked directly to review the RL layer itself — traced the actual training environment (`physics/lob_env.py`), not just the execution layer, and found the *same* long-only bug one level deeper, plus a second, independent mismatch.

**The long-only bug existed in the training signal itself, not just execution.** `TradeJackLOBEnv.step()` had no clamp at all on `qty_delta` — every model was trained in an environment that permitted "selling" more than it held, simulating short positions no real spot account could ever take. Fixing this only in `execution/paper_exchange.py`/`live_exchange_bridge.py` (done last round) was necessary but insufficient: those fixes stop an impossible order from *executing*, but the model was still being *trained* to want something it can never have. Fixed identically at the source: any target implying a short is now clamped to fully exiting the position, never further. Verified in the actual training env, not a mock: went long, then hit it with an extreme short target, confirmed position stayed at exactly zero rather than going negative.

**A second, independent train/serve mismatch, in trading fees**: the training environment assumed a 0.04% taker fee — less than half of `execution/paper_exchange.py`'s (and Binance's actual verified VIP0 default) 0.10%. Every model was being trained against costs cheaper than what it will actually pay in paper or live execution, meaning training-time performance estimates were systematically optimistic. Fixed to match.

**Researched reward function design properly before touching it, rather than guessing.** Confirmed the Differential Sharpe Ratio (Moody & Wu, 1997) remains a live, cited approach in current (2026) RL-for-trading literature — but the evidence is genuinely mixed: at least one recent direct comparison found training against the *exact* Sharpe ratio outperformed the *differential* (online-approximated) version on the same task. Given that split evidence, and that the existing `log_ret - dd_penalty` reward already has some risk-adjustment built in, the DSR was added as an **opt-in alternative** (`reward_mode="differential_sharpe"`), not a replacement of the default — implemented following the exact online-update formula from a 2026 RL-trading-environment paper, with per-episode state (`_dsr_mu`, `_dsr_m2`) correctly reset at `reset()` so it doesn't leak across episodes. Verified numerically stable (no NaN/Inf) over 80 random-action steps, and confirmed the untouched default reward mode still produces identical behavior to before.

Full tournament regression (2×PPO, SAC, DQN) re-verified clean after all three fixes.

**Reviewed but deliberately not changed, with reasoning stated**: the action space stays `Box(-1, 1)` rather than being rescaled to `Box(0, 1)` to match the long-only constraint exactly. Rescaling would eliminate the wasted policy capacity in the never-executable negative half, but breaks compatibility with every existing trained checkpoint's fitted output distribution and would require fresh training runs across all architectures — a real, worthwhile upgrade to schedule deliberately, not a hot patch to force through alongside a bug fix. Also reviewed the Kyle's-Lambda execution-friction model (`compute_friction_fill_price`) — a smooth, differentiable slippage proxy rather than full order-book simulation, which is standard and reasonable for RL training specifically (real book-walking is noisier and non-differentiable); no change made.

---

## 🔬 14. Reward & Penalty Review for Generalization (v3.7)

Asked directly to review the reward/penalty structure specifically for generalization — found one more real, previously-undiscovered bug with direct research backing for why it matters, plus two new opt-in, research-grounded reward components.

**Found: the taker fee was scaled by total position size, not trade size.** `notional_value` (used to compute the per-trade fee) was computed from `self.position_qty` *after* it had already been updated to the new target — meaning the fee for adjusting a large existing position by even 1% was charged as if the *entire* position had just been traded. Verified directly: a ~90% position entry consumed $9,009 in a $10,000 account (fee correctly proportional to that large trade), while a subsequent tiny 1% rebalance on top of it consumed only $97 — previously, that tiny rebalance would have been charged a fee sized to the *entire* ~91% position, not the 1% actually traded. This connects directly to current research: "realistic evaluation must penalize turnover and execution cost, as methods that ignore these often overfit" is a recurring finding across multiple 2020–2026 RL-for-trading papers — a fee signal that doesn't scale with actual trade size teaches exactly the wrong lesson about transaction costs, in a way that specifically degrades out-of-sample generalization rather than just being numerically off. Fixed to scale with the real traded notional; funding/carry cost (which *should* scale with total position held, unlike a trade fee) was left as-is since that part was already correct.

**Added two new opt-in reward components**, both defaulting to exactly zero effect (verified: reward is float-identical whether their coefficients are left as defaults or passed explicitly as `0.0`) so nothing about existing trained checkpoints' training signal changes retroactively:
- `turnover_penalty_coef`: penalizes traded notional relative to equity, separate from the fee itself — the literature's "transaction burden" component, addressing that a raw fee may not be large enough at the margin a gradient-based policy actually explores to discourage overtrading on noise.
- `continuous_risk_penalty_coef`: a dense, every-tick penalty proportional to *current* drawdown level, supplementing the existing sparse penalty that only fires on new high-water-mark breaches. The existing `dd_penalty` alone gives near-zero risk-aversion gradient for long stretches then spikes sharply — a known source of high gradient variance; several risk-sensitive-RL papers (CVaR-based shaping, quadratic risk terms in portfolio-RL) use a continuous term for exactly this reason.

Both verified independently: turnover penalty measurably reduces reward on an actual trade when enabled; continuous risk penalty remains numerically stable (no NaN/Inf) across 40 random-action steps. Full tournament regression re-confirmed clean after all changes.

**Explicitly not added, with reasoning**: an "overtrading frequency" penalty distinct from the turnover-magnitude penalty above (some literature separates "how much you traded" from "how often you traded") — the existing `min_hold_ticks` mechanism in `execution/risk_guardian.py` already addresses trade frequency at the execution layer, and duplicating that constraint inside the training reward risked fighting itself against a mechanism that already works. One risk regularizer per concern, not two overlapping ones.
