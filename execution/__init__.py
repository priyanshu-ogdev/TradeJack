"""
execution/ — Live market connection & paper-trading validation layer.

This module answers one question honestly, using real market data and a synthetic
wallet: "if this model's weights were frozen and released against the live market
right now, what would actually happen?" No third-party demo-broker account is used
(RoboForex-style simulated accounts were considered and rejected — see
binance_live_feed.py docstring for why). Instead this module streams Binance's real
public order book directly and simulates the wallet, fees, and latency in-process.

Components:
    binance_live_feed.py   Real Binance WebSocket L2 depth stream (read-only, no API
                            key required). Reuses data_forge.lob_collector.LocalOrderBook
                            for the order-book reconstruction so there is exactly one
                            implementation of Binance's local-book protocol in the repo.
    streaming_features.py  Incremental approximation of the OFI / VPIN / Kyle's-lambda
                            style features the models were trained on. Flagged explicitly
                            as an approximation — see its docstring for the train/serve
                            skew risk this introduces.
    paper_exchange.py       Synthetic wallet that fills orders by walking the REAL live
                            order book (not a formula), with realistic latency, fees,
                            and exchange filters. Reuses physics.portfolio_tracker for
                            equity/ledger bookkeeping instead of reinventing it.
    risk_guardian.py        Pre-trade safety layer: staleness checks, position/rate
                            limits, daily-loss and drawdown halts, kill switch.
    live_inference_server.py  Wires the above into one continuous loop: feed -> features
                            -> frozen model -> risk guardian -> paper exchange -> ledger.
    session_report.py       Turns a session's ledger into a P&L / Sharpe / Sortino /
                            fee-drag / vs-buy-and-hold report, with a significance test —
                            the actual answer to "how much could this earn if released".

Explicitly out of scope here (by instruction): data_forge/ and tests/ are not modified.
Nothing in this module places a real order — DEPLOY_CONFIG.auto_promotion and any real
exchange key wiring stay untouched until that is explicitly asked for separately.
"""
