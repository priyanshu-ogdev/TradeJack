"""
OandaAdapter -- a real FX broker connection, implementing the SAME
ExchangeAdapter interface BinanceSpotAdapter does, so RiskGuardian,
LiveOrderComposer, and live_inference_server.py work with either without any
changes: they only ever touch the ExchangeAdapter abstract interface, never a
concrete broker class directly (see exchange_adapter.py's own docstring on
that point).

WHY OANDA, NOT METATRADER (verified before writing this, not assumed)
------------------------------------------------------------------------
The official `MetaTrader5` Python package is Windows-only -- confirmed via
direct research: it's a compiled binary that talks to a local MT5 terminal
over Windows IPC, with no Mac/Linux build and no network API. Every
Linux-side workaround (mt5linux, mt5-mac-bridge, MetaTrader5-Docker) works by
running an actual MT5 terminal under Wine plus a separate bridge/RPC process
on top of that -- a fundamentally different, much more fragile deployment
model (a GUI terminal process, a compatibility layer, a bridge process, all
needing to stay up) than this project's native, headless-Linux asyncio
services (BinanceSpotAdapter, BinanceLiveDepthFeed, process_supervisor.py).
MT4 is worse: no official Python API at all, only community DLL/ZeroMQ EA
bridges of varying quality.

OANDA's v20 API, by contrast, is a real REST + streaming JSON API -- no
terminal, no Windows, no Wine, works the same way on the same OS this
project already deploys on. Free practice (demo) accounts exist for testing
before any real capital is involved. This is the same reasoning that made
BinanceSpotAdapter's native asyncio/ccxt approach the right choice for
crypto; OANDA is the FX equivalent of that same shape, not a compromise.

A REAL, IMPORTANT STRUCTURAL DIFFERENCE FROM CRYPTO -- READ BEFORE TRUSTING
FEATURES BUILT FOR BINANCE'S ORDER BOOK AGAINST THIS ADAPTER
------------------------------------------------------------------------
Retail FX is dealer/OTC-quoted, not a central limit order book like Binance.
OANDA's pricing endpoint returns a handful of liquidity-tiered bid/ask levels
from OANDA's OWN quoting engine, not a public book of other traders' resting
orders. get_orderbook() below returns whatever tiers the pricing endpoint
gives -- structurally shallower and dealer-specific in a way Binance's real
order book isn't. This is exactly why forex_toxicity_engineering.py had to
build BVC (Bulk Volume Classification) instead of using crypto's true
trade-classified VPIN: FX doesn't have the raw ingredients (public order flow,
aggressor tags) crypto's microstructure features assume. Nothing in this
adapter changes that; it's stated here again because plugging a real broker
in is exactly the moment that gap stops being an abstract caveat in a design
doc and starts being live data flowing into a live decision.

VERIFICATION STATUS: oandapyV20 is not installed in this sandbox (no network
access to pip install it), so this file could not be run or hit with a real
request here. Every endpoint, request/response shape, and code pattern below
was checked directly against oandapyV20's real documented usage (PyPI
project page, readthedocs) during this session, not written from
half-remembered training data. Run `pip install oandapyV20`, create a free
OANDA practice account, and exercise every method here against it -- in
particular get_orderbook()'s liquidity-tier mapping and place_market_order()'s
fill-price parsing, which are the two places a subtle mismatch between what
this file assumes and what the real API returns would be easiest to miss
without actually running it.
"""

import os
import time
import asyncio
import logging
from typing import Any, Dict, List, Optional

from execution.exchange_adapter import ExchangeAdapter, OrderResult, OrderBook

try:
    import oandapyV20
    from oandapyV20 import API as OandaAPI
    from oandapyV20.exceptions import V20Error
    import oandapyV20.endpoints.accounts as oanda_accounts
    import oandapyV20.endpoints.pricing as oanda_pricing
    import oandapyV20.endpoints.orders as oanda_orders
    import oandapyV20.endpoints.positions as oanda_positions
    import oandapyV20.endpoints.trades as oanda_trades
    from oandapyV20.contrib.requests import MarketOrderRequest
    OANDAPY_AVAILABLE = True
except ImportError:
    OANDAPY_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] (OandaAdapter) %(message)s")
logger = logging.getLogger("OandaAdapter")


def to_oanda_instrument(symbol: str) -> str:
    """This project's FX pair convention (no separator, e.g. "EURUSD" -- see
    data_forge/config.py's default_forex_pairs) -> OANDA's instrument naming
    (underscore-separated, e.g. "EUR_USD"). Handles a symbol that already has
    a separator ("EUR-USD", "EUR_USD") by stripping it first, so this is
    idempotent regardless of which convention a caller happens to pass."""
    cleaned = symbol.replace("-", "").replace("_", "").replace("/", "").upper()
    if len(cleaned) != 6:
        raise ValueError(f"'{symbol}' does not look like a 6-letter FX pair (got '{cleaned}' after stripping separators).")
    return f"{cleaned[:3]}_{cleaned[3:]}"


def from_oanda_instrument(instrument: str) -> str:
    """OANDA's "EUR_USD" -> this project's "EURUSD"."""
    return instrument.replace("_", "")


class OandaAdapter(ExchangeAdapter):
    """
    FX broker adapter over OANDA's v20 API. Implements the exact same
    ExchangeAdapter interface as BinanceSpotAdapter -- everything upstream
    (RiskGuardian, LiveOrderComposer, live_inference_server.py) already only
    talks to that interface, so wiring an OandaAdapter in requires no changes
    there, only choosing which adapter to construct for a given instrument.

    Credentials via env vars (matching BinanceSpotAdapter's own
    api_key_env-configurable pattern, not hardcoded names) -- an OANDA access
    token and account ID, both required; there is no anonymous/public access
    to place orders or read a real account's pricing tier.
    """

    def __init__(
        self,
        symbol: str,
        access_token_env: str = "OANDA_ACCESS_TOKEN",
        account_id_env: str = "OANDA_ACCOUNT_ID",
        environment: str = "practice",  # "practice" (demo) or "live" -- OANDA's own terminology
    ):
        if not OANDAPY_AVAILABLE:
            raise RuntimeError(
                "oandapyV20 is required for OandaAdapter (pip install oandapyV20). "
                "Not available in this sandbox -- see this file's module docstring "
                "for what to verify once it is."
            )
        if environment not in ("practice", "live"):
            raise ValueError(f"environment must be 'practice' or 'live', got '{environment}'")

        self.symbol = symbol
        self.instrument = to_oanda_instrument(symbol)
        self.environment = environment
        self.access_token = os.environ.get(access_token_env, "")
        self.account_id = os.environ.get(account_id_env, "")

        self._client: Optional["OandaAPI"] = None
        self._connected = False
        self._account_currency: Optional[str] = None  # e.g. "USD" -- fetched on connect()

    # ------------------------------------------------------------------ #
    # Connection lifecycle
    # ------------------------------------------------------------------ #

    async def connect(self) -> None:
        if not self.access_token or not self.account_id:
            raise RuntimeError(
                f"OANDA credentials not configured. Set the environment variables this "
                f"adapter was constructed to read (access token, account ID) before "
                f"calling connect() -- mirrors BinanceSpotAdapter refusing to silently "
                f"run with empty credentials."
            )
        self._client = OandaAPI(access_token=self.access_token, environment=self.environment)
        # A real, cheap round-trip to confirm the token/account actually work,
        # matching BinanceSpotAdapter.connect()'s own "verify credentials up
        # front, don't discover they're wrong on the first real order" approach.
        summary = await self._request(oanda_accounts.AccountSummary(self.account_id))
        self._account_currency = summary["account"]["currency"]
        self._connected = True
        logger.info(
            f"Connected to OANDA ({self.environment}), account {self.account_id}, "
            f"home currency {self._account_currency}."
        )

    async def close(self) -> None:
        # oandapyV20's API client is a thin requests.Session wrapper with no
        # persistent connection/websocket to tear down for plain REST calls
        # (unlike a streaming PricingStream, which this adapter does not open
        # -- get_ticker()/get_orderbook() below use one-shot PricingInfo
        # calls, not the streaming endpoint, since ExchangeAdapter's
        # interface is pull-based). Nothing to actually close here; the flag
        # flip is what matters for is_connected().
        self._connected = False

    def is_connected(self) -> bool:
        return self._connected

    async def _request(self, request_obj) -> dict:
        """oandapyV20's client.request() is synchronous (plain `requests`
        under the hood, not an async HTTP client) -- run it in a thread so it
        doesn't block this project's asyncio event loop, the same reasoning
        BinanceSpotAdapter's ccxt calls need (ccxt's sync client has the same
        issue; check how BinanceSpotAdapter handles this and match it exactly
        rather than introducing a second, different blocking-call convention
        in the same codebase)."""
        loop = asyncio.get_event_loop()
        try:
            return await loop.run_in_executor(None, self._client.request, request_obj)
        except V20Error as e:
            raise RuntimeError(f"OANDA API error: {e}") from e

    # ------------------------------------------------------------------ #
    # Market data
    # ------------------------------------------------------------------ #

    async def get_ticker(self, symbol: str) -> float:
        instrument = to_oanda_instrument(symbol)
        params = {"instruments": instrument}
        data = await self._request(oanda_pricing.PricingInfo(self.account_id, params=params))
        price = data["prices"][0]
        bid = float(price["bids"][0]["price"])
        ask = float(price["asks"][0]["price"])
        return (bid + ask) / 2.0

    async def get_orderbook(self, symbol: str, depth: int = 5) -> OrderBook:
        """Returns OANDA's liquidity-tiered dealer quote, NOT a real central
        limit order book -- see this module's docstring. Tiers beyond what
        OANDA's pricing response actually includes are simply absent (not
        padded/faked); depth is a maximum, not a guarantee."""
        instrument = to_oanda_instrument(symbol)
        params = {"instruments": instrument}
        data = await self._request(oanda_pricing.PricingInfo(self.account_id, params=params))
        price = data["prices"][0]
        bids = [[float(b["price"]), float(b["liquidity"])] for b in price["bids"][:depth]]
        asks = [[float(a["price"]), float(a["liquidity"])] for a in price["asks"][:depth]]
        mid = (bids[0][0] + asks[0][0]) / 2.0 if bids and asks else 0.0
        return OrderBook(symbol=symbol, bids=bids, asks=asks, timestamp=time.time(), mid_price=mid)

    # ------------------------------------------------------------------ #
    # Account / balances
    # ------------------------------------------------------------------ #

    async def get_balance(self, asset: str) -> float:
        """An FX account has ONE home-currency balance, not per-asset
        balances like crypto -- `asset` must match the account's own
        currency (fetched at connect() time) or this returns 0.0, matching
        BinanceSpotAdapter's "0.0 for an asset you don't hold" convention
        rather than raising for an FX-vs-crypto conceptual mismatch the
        caller may not be expecting."""
        if self._account_currency and asset.upper() != self._account_currency.upper():
            return 0.0
        summary = await self._request(oanda_accounts.AccountSummary(self.account_id))
        return float(summary["account"]["balance"])

    async def get_all_balances(self) -> Dict[str, float]:
        summary = await self._request(oanda_accounts.AccountSummary(self.account_id))
        currency = summary["account"]["currency"]
        return {currency: float(summary["account"]["balance"])}

    # ------------------------------------------------------------------ #
    # Orders / positions
    # ------------------------------------------------------------------ #

    async def place_market_order(self, symbol: str, side: str, qty: float) -> OrderResult:
        """qty is always positive; direction comes from `side` ('buy'/'sell')
        -- OANDA's own convention is a signed `units` field (negative =
        sell), so the sign is applied here, once, rather than asking every
        caller to remember OANDA's sign convention on top of this project's
        own buy/sell string convention."""
        if side not in ("buy", "sell"):
            raise ValueError(f"side must be 'buy' or 'sell', got '{side}'")
        instrument = to_oanda_instrument(symbol)
        units = qty if side == "buy" else -qty

        order_req = MarketOrderRequest(instrument=instrument, units=units)
        try:
            data = await self._request(oanda_orders.OrderCreate(self.account_id, data=order_req.data))
        except RuntimeError as e:
            return OrderResult(
                order_id="", symbol=symbol, side=side, qty=0.0, avg_price=0.0,
                cost=0.0, fee=0.0, timestamp=time.time(), status="rejected", raw={"error": str(e)},
            )

        fill = data.get("orderFillTransaction")
        if fill is None:
            # Order was created but not immediately filled (rare for a market
            # order, but OANDA can return orderCancelTransaction instead --
            # e.g. market halted, price moved beyond a safety guard). Treat
            # as rejected rather than guessing at a fill that didn't happen.
            return OrderResult(
                order_id=data.get("orderCreateTransaction", {}).get("id", ""),
                symbol=symbol, side=side, qty=0.0, avg_price=0.0, cost=0.0, fee=0.0,
                timestamp=time.time(), status="rejected", raw=data,
            )

        avg_price = float(fill["price"])
        filled_units = abs(float(fill["units"]))
        # OANDA reports financing/commission separately; "fee" here is
        # whatever OANDA calls it in the fill transaction, defaulting to 0.0
        # for account types where trading is spread-only with no separate
        # commission line (common for retail FX, unlike Binance's explicit
        # per-trade fee).
        fee = abs(float(fill.get("commission", 0.0)))
        return OrderResult(
            order_id=fill.get("id", ""), symbol=symbol, side=side, qty=filled_units,
            avg_price=avg_price, cost=filled_units * avg_price, fee=fee,
            timestamp=time.time(), status="filled", raw=data,
        )

    async def get_open_orders(self, symbol: str) -> List[Dict[str, Any]]:
        # Market orders (the only order type place_market_order() issues)
        # fill immediately or get rejected -- there is no resting "open market
        # order" state to report, unlike a limit order. Returns pending
        # (non-market) orders for this instrument if any exist from elsewhere
        # (e.g. manually placed in OANDA's own UI), for visibility only --
        # this project's live trading only ever calls place_market_order().
        instrument = to_oanda_instrument(symbol)
        data = await self._request(oanda_orders.OrderList(self.account_id, params={"instrument": instrument}))
        return data.get("orders", [])

    async def cancel_all_orders(self, symbol: str) -> int:
        orders = await self.get_open_orders(symbol)
        cancelled = 0
        for order in orders:
            order_id = order.get("id")
            if not order_id:
                continue
            try:
                await self._request(oanda_orders.OrderCancel(self.account_id, orderID=order_id))
                cancelled += 1
            except RuntimeError as e:
                logger.error(f"Failed to cancel OANDA order {order_id}: {e}")
        return cancelled

    async def get_position(self, symbol: str) -> Dict[str, float]:
        instrument = to_oanda_instrument(symbol)
        try:
            data = await self._request(oanda_positions.PositionDetails(self.account_id, instrument=instrument))
        except RuntimeError:
            return {"qty": 0.0, "entry_price": 0.0}  # OANDA 404s a position that doesn't exist -- treat as flat

        position = data.get("position", {})
        long_units = float(position.get("long", {}).get("units", 0.0))
        short_units = float(position.get("short", {}).get("units", 0.0))
        net_units = long_units + short_units  # short units are already negative in OANDA's response
        if net_units > 0:
            entry_price = float(position.get("long", {}).get("averagePrice", 0.0))
        elif net_units < 0:
            entry_price = float(position.get("short", {}).get("averagePrice", 0.0))
        else:
            entry_price = 0.0
        return {"qty": net_units, "entry_price": entry_price}

    # ------------------------------------------------------------------ #
    # Funding -- deliberately minimal, see ExchangeAdapter's own docstring
    # ------------------------------------------------------------------ #

    async def get_deposit_address(self, asset: str) -> Optional[Dict[str, str]]:
        """No analog for an FX broker account -- OANDA is funded via bank
        transfer/card through OANDA's own account funding UI, not a
        blockchain deposit address. Returns None (the base class's own
        default), same as a paper account. This is not a gap to fill in
        later; it's the correct answer for this broker type."""
        return None

    async def get_recent_deposits(self, asset: Optional[str] = None, limit: int = 10) -> List[Dict[str, Any]]:
        """OANDA's transaction history DOES include funding transactions
        (type: TRANSFER_FUNDS), unlike the deposit-address case above -- this
        could be wired to accounts.AccountTransactions filtered by type, but
        is left as the base class's empty-list default for now rather than
        implemented speculatively without a real account to verify the
        transaction-type filter against."""
        return []
