"""
Execution module — Live market connection, risk management, and order routing.

Components:
  - exchange_adapter: Abstract exchange interface + Binance Spot implementation (ccxt)
  - paper_exchange: Simulated exchange using LOB physics for paper trading
  - risk_guardian: Immutable safety layer between model and exchange
  - live_inference_server: Production inference loop (frozen model + live data)
  - position_throttle: Rolling-Sortino-based position sizing (replaces live burn)
"""
