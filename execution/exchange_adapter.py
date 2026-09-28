"""
Exchange Adapter — Abstract exchange interface + Binance Spot implementation.

Uses ccxt for Binance API interaction. Supports testnet and mainnet via config.
Secrets loaded from environment variables (never stored in code).

RESEARCHED, NOT ASSUMED (Binance's own API docs and changelog, checked because
this evolves and a training cutoff can't be trusted for exchange-integration
specifics):
  - Weight limit is 6,000/minute per IP (raised from 1,200 in Aug 2023) and
    the RAW_REQUESTS limit is being raised further from 2026-04-02. Binance's
    own rate-limit docs explicitly say: "Please use WebSocket Streams for
    live updates to avoid bans" — polling REST for things a stream could push
    is the thing that gets IPs banned, not just a style preference.
  - HTTP 429 means back off (a `Retry-After`/`retryAfter` is provided);
    HTTP 418 means the IP is ALREADY BANNED, for a duration that "scales for
    repeat offenders, from 2 minutes to 3 days" — continuing to hit the API
    during a ban makes it worse. These two cases need different handling,
    not one generic "order failed" catch-all.
  - As of 2026-02-20 07:00 UTC, Binance discontinued the REST `listenKey`
    mechanism for Spot user data streams entirely (`POST/PUT/DELETE
    /api/v3/userDataStream`). The replacement is subscribing to the user
    data stream directly through the WebSocket API
    (`userDataStream.subscribe`, a signed WS API request) — NOT a REST
    listenKey you poll/keepalive. This adapter does not implement real-time
    order/balance push notifications at all yet (see the class docstring
    below for why that's a deliberate, stated gap rather than a rushed
    implementation) — if you build it, build it against this current
    mechanism. Do not implement or copy example code using `listenKey`/
    `userDataStream.start` for Spot; that mechanism no longer exists.

Usage:
    adapter = BinanceSpotAdapter(testnet=True)
    await adapter.connect()
    balance = await adapter.get_balance("USDT")
    result = await adapter.place_market_order("BTC/USDT", "buy", 0.0001)
    await adapter.close()
"""

import os
import time
import logging
import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] (ExchangeAdapter) %(message)s")
logger = logging.getLogger("ExchangeAdapter")

try:
    import ccxt.async_support as ccxt_async
    import ccxt
    CCXT_AVAILABLE = True
except ImportError:
    CCXT_AVAILABLE = False
    logger.warning("ccxt not installed. Exchange adapters unavailable. pip install ccxt")


class ExchangeBannedError(Exception):
    """Raised when the exchange has IP-banned this connection (HTTP 418).
    Distinct from a generic order rejection on purpose — callers (RiskGuardian)
    should treat this as a hard, time-boxed halt, not "this one order failed,
    try the next signal." """
    def __init__(self, message: str, retry_after_ts: Optional[float] = None):
        super().__init__(message)
        self.retry_after_ts = retry_after_ts


class BinanceRateLimitTracker:
    """
    Tracks Binance's actual server-reported rate-limit usage from response
    headers, per Binance's own documented pattern (X-MBX-USED-WEIGHT-*,
    X-MBX-ORDER-COUNT-*). ccxt's `enableRateLimit: True` already paces
    requests against its own static per-endpoint weight table, which is a
    reasonable baseline — this adds the officially-recommended layer on top:
    reading what the SERVER actually reports back, which is the only source
    of truth if anything else (another process, a shared IP) is also
    consuming the same budget.
    """

    def __init__(self, weight_limit_per_minute: int = 6000, throttle_threshold: float = 0.8):
        self.weight_limit = weight_limit_per_minute
        self.throttle_threshold = throttle_threshold
        self.last_used_weight: Optional[int] = None
        self.banned_until_ts: Optional[float] = None

    def update_from_headers(self, headers: Optional[Dict[str, str]]):
        if not headers:
            return
        for key, value in headers.items():
            if key.upper().startswith("X-MBX-USED-WEIGHT"):
                try:
                    self.last_used_weight = int(value)
                except (TypeError, ValueError):
                    pass

    def is_banned(self) -> bool:
        return self.banned_until_ts is not None and time.time() < self.banned_until_ts

    def register_ban(self, retry_after_ts: Optional[float]):
        # retryAfter from Binance is typically an epoch-ms timestamp, not a duration.
        if retry_after_ts and retry_after_ts > time.time():
            self.banned_until_ts = retry_after_ts
        else:
            self.banned_until_ts = time.time() + 120  # conservative floor: Binance's own minimum ban is 2 minutes
        logger.critical(
            f"BINANCE IP BAN (HTTP 418) — halting ALL exchange calls until "
            f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.banned_until_ts))}. "
            f"Continuing to call during a ban extends it — this is a hard stop, not a retry-soon."
        )

    def should_throttle(self) -> bool:
        if self.last_used_weight is None:
            return False
        return (self.last_used_weight / self.weight_limit) >= self.throttle_threshold

    def usage_fraction(self) -> float:
        if self.last_used_weight is None:
            return 0.0
        return self.last_used_weight / self.weight_limit


@dataclass
class OrderResult:
    """Result of an order placement."""
    order_id: str
    symbol: str
    side: str           # "buy" or "sell"
    qty: float          # Filled quantity
    avg_price: float    # Average fill price
    cost: float         # Total cost in quote currency
    fee: float          # Fee paid
    timestamp: float    # Epoch seconds
    status: str         # "filled", "partial", "rejected", "banned", "invalid"
    raw: Dict[str, Any] = None


@dataclass
class OrderBook:
    """Snapshot of limit order book."""
    symbol: str
    bids: List[List[float]]   # [[price, qty], ...]
    asks: List[List[float]]   # [[price, qty], ...]
    timestamp: float
    mid_price: float


class ExchangeAdapter(ABC):
    """
    Abstract exchange adapter interface.

    All exchange implementations (Binance, paper, testnet) implement this.
    The LiveInferenceServer and RiskGuardian operate purely through this interface,
    ensuring the model never directly touches exchange-specific code.
    """

    @abstractmethod
    async def connect(self) -> None:
        """Establish connection to exchange."""

    @abstractmethod
    async def close(self) -> None:
        """Clean up connections."""

    @abstractmethod
    async def get_ticker(self, symbol: str) -> float:
        """Get current mid-price for symbol."""

    @abstractmethod
    async def get_orderbook(self, symbol: str, depth: int = 5) -> OrderBook:
        """Get L2 order book snapshot."""

    @abstractmethod
    async def get_balance(self, asset: str) -> float:
        """Get available balance for a specific asset."""

    @abstractmethod
    async def get_all_balances(self) -> Dict[str, float]:
        """Get all non-zero balances."""

    @abstractmethod
    async def place_market_order(self, symbol: str, side: str, qty: float) -> OrderResult:
        """Place a market order. side: 'buy' or 'sell'."""

    @abstractmethod
    async def get_open_orders(self, symbol: str) -> List[Dict[str, Any]]:
        """Get all open orders for symbol."""

    @abstractmethod
    async def cancel_all_orders(self, symbol: str) -> int:
        """Cancel all open orders for symbol. Returns count cancelled."""

    @abstractmethod
    async def get_position(self, symbol: str) -> Dict[str, float]:
        """Get current position for symbol. Returns {'qty': float, 'entry_price': float}."""

    async def get_deposit_address(self, asset: str) -> Optional[Dict[str, str]]:
        """
        Returns {'address': str, 'network': str, 'tag': Optional[str]} for
        depositing `asset` — i.e. where YOU send funds from your bank/wallet
        to fund this account. Default implementation returns None; only
        BinanceSpotAdapter overrides this with a real answer, since a paper
        account has no real deposit address.

        Deliberately no `withdraw()` method exists anywhere in this interface
        or its implementations, despite ccxt supporting it. Depositing is a
        read-only lookup with no way to move funds OUT; withdrawal is a
        genuinely dangerous capability to expose from a dashboard or any
        automated system, and nothing in this project needs a bot that can
        pull money out of the account it's trading with. If you need to
        withdraw, do it directly on Binance's own site/app.
        """
        return None

    async def get_recent_deposits(self, asset: Optional[str] = None, limit: int = 10) -> List[Dict[str, Any]]:
        """Returns recent deposit history — lets a dashboard show 'funds
        received' without the bot ever needing withdrawal capability to do
        so. Default implementation returns an empty list."""
        return []

    @abstractmethod
    def is_connected(self) -> bool:
        """Check if exchange connection is healthy."""


class BinanceSpotAdapter(ExchangeAdapter):
    """
    Binance Spot exchange adapter via ccxt.

    Supports both testnet (testnet.binance.vision) and mainnet (api.binance.com).
    API keys loaded from environment variables specified in DeploymentConfig.

    Rate limiting: ccxt's own `enableRateLimit: True` paces requests against a
    static per-endpoint weight table; `BinanceRateLimitTracker` (above) adds
    the officially-recommended layer of reading actual server-reported usage
    from response headers, and hard-stops all calls for the real ban duration
    on HTTP 418 rather than treating it like any other failed request.

    DELIBERATE GAP, stated rather than hidden: no real-time order/balance push
    notifications (user data stream) are implemented. Binance's own guidance
    is to prefer streams over REST polling for live updates, and this adapter
    is pure REST request/response for every method. This is a more defensible
    gap than it sounds for a market-order-only, rate-limited execution loop
    (a market order's REST response includes its fill details synchronously,
    unlike a resting limit order that genuinely needs a push to know when it
    eventually fills) — but it means there's no independent reconciliation
    signal if a connection drops mid-request, and it does mean this adapter
    consumes more request weight than necessary for balance/position checks
    that a stream would push for free. Implementing this properly means
    subscribing via the WebSocket API's `userDataStream.subscribe` (signed),
    NOT the deprecated REST listenKey — see the module docstring. Not
    implemented here because hand-rolling a signed WS API session without any
    way to test it against a real Binance connection from this environment is
    exactly the kind of real-money-adjacent code that's worth building
    carefully with real connectivity to verify against, not rushing.
    """

    # Binance Spot testnet URLs
    TESTNET_URL = "https://testnet.binance.vision"
    TESTNET_WS = "wss://testnet.binance.vision/ws"

    def __init__(
        self,
        testnet: bool = True,
        api_key_env: str = "BINANCE_API_KEY",
        api_secret_env: str = "BINANCE_API_SECRET",
        ed25519_key_path_env: str = "BINANCE_ED25519_PRIVATE_KEY_PATH",
    ):
        """
        Credential loading, in order of preference (researched against
        Binance's own current API docs, not assumed):

        1. Ed25519 private key file (path in `ed25519_key_path_env`) — Binance's
           own docs now say plainly "HMAC keys are deprecated. We recommend to
           migrate to asymmetric API keys, such as Ed25519" and "It is highly
           recommended to use Ed25519 API keys as it should provide the best
           performance and security out of all supported key types" (smaller,
           faster-to-verify signatures than RSA; no shared secret transmitted
           at all, unlike HMAC). Generate one with Binance's own Asymmetric
           Keys Generator tool, register the PUBLIC key on Binance, and point
           this at the PRIVATE key's PEM file — never commit that file.
        2. Legacy HMAC secret (`api_secret_env`) — still functional (Binance
           has not announced a hard removal date as of this writing), but
           logs a deprecation warning every time it's the only credential
           available, since Binance's own guidance is to migrate off it.

        No ccxt changes needed for this — verified by reading ccxt's actual
        binance.py signing code: it inspects whether `self.secret` contains
        the string "PRIVATE KEY" and automatically switches to RSA or Ed25519
        signing (by PEM length) instead of HMAC. Passing a loaded Ed25519 PEM
        string as ccxt's `secret` config value is the entire integration.
        """
        if not CCXT_AVAILABLE:
            raise ImportError("ccxt required for BinanceSpotAdapter. pip install ccxt")

        self.testnet = testnet
        self.api_key = os.environ.get(api_key_env, "")
        self.signing_mode = "none"
        self.secret_value = ""

        ed25519_path = os.environ.get(ed25519_key_path_env, "")
        if ed25519_path and os.path.exists(ed25519_path):
            with open(ed25519_path, "r") as f:
                pem_content = f.read().strip()
            if "PRIVATE KEY" not in pem_content:
                logger.error(
                    f"'{ed25519_path}' does not look like a PEM private key (no 'PRIVATE KEY' marker) — "
                    "falling back to HMAC secret if available."
                )
            else:
                self.secret_value = pem_content
                self.signing_mode = "ed25519_or_rsa"  # ccxt itself distinguishes by PEM length at request-signing time
                logger.info(f"Loaded asymmetric signing key from '{ed25519_path}' (Ed25519/RSA — Binance's currently recommended key type).")

        if self.signing_mode == "none":
            legacy_secret = os.environ.get(api_secret_env, "")
            if legacy_secret:
                self.secret_value = legacy_secret
                self.signing_mode = "hmac"
                logger.warning(
                    "Using legacy HMAC API secret. Binance's own docs: 'HMAC keys are deprecated. "
                    "We recommend to migrate to asymmetric API keys, such as Ed25519.' Consider generating "
                    f"an Ed25519 key pair and setting {ed25519_key_path_env} instead."
                )

        self.api_secret = self.secret_value  # kept for backward-compat attribute access elsewhere
        self.exchange: Optional[ccxt_async.binance] = None
        self._connected = False
        self._last_heartbeat = 0.0
        self.rate_limiter = BinanceRateLimitTracker()

        if not self.api_key or not self.secret_value:
            logger.warning(
                f"Exchange credentials not found (checked {ed25519_key_path_env} and {api_secret_env}). "
                "Set one of them in .env. Orders will fail."
            )

    async def connect(self) -> None:
        """Initialize ccxt async Binance exchange."""
        config = {
            "apiKey": self.api_key,
            "secret": self.api_secret,
            "enableRateLimit": True,
            "options": {"defaultType": "spot"},
        }

        if self.testnet:
            config["urls"] = {
                "api": {
                    "public": self.TESTNET_URL + "/api/v3",
                    "private": self.TESTNET_URL + "/api/v3",
                },
            }
            # Also set via sandbox mode for ccxt
            config["sandbox"] = True

        self.exchange = ccxt_async.binance(config)

        if self.testnet:
            self.exchange.set_sandbox_mode(True)

        # Verify connectivity
        try:
            await self.exchange.load_markets()
            self._connected = True
            self._last_heartbeat = time.time()
            self.rate_limiter.update_from_headers(getattr(self.exchange, "last_response_headers", None))
            mode = "TESTNET" if self.testnet else "MAINNET"
            logger.info(f"Binance Spot adapter connected ({mode}). Markets loaded: {len(self.exchange.markets)}")
        except Exception as e:
            self._connected = False
            logger.error(f"Failed to connect to Binance: {e}")
            raise

    async def close(self) -> None:
        if self.exchange:
            await self.exchange.close()
            self._connected = False
            logger.info("Binance Spot adapter disconnected.")

    def is_connected(self) -> bool:
        return self._connected and (time.time() - self._last_heartbeat < 120)

    def _post_call_bookkeeping(self):
        """Call after every ccxt request: updates the rate-limit tracker from
        whatever headers ccxt captured, and refreshes the heartbeat."""
        self._last_heartbeat = time.time()
        headers = getattr(self.exchange, "last_response_headers", None)
        self.rate_limiter.update_from_headers(headers)
        if self.rate_limiter.should_throttle():
            logger.warning(
                f"Approaching Binance rate limit ({self.rate_limiter.usage_fraction()*100:.0f}% of "
                f"{self.rate_limiter.weight_limit}/min used, server-reported) — consider reducing call frequency."
            )

    def _check_not_banned(self):
        """Raises ExchangeBannedError if we're inside a known ban window,
        without making any network call — the whole point of tracking bans
        locally is to stop hitting a banned IP, not to find out we're still
        banned by trying again."""
        if self.rate_limiter.is_banned():
            raise ExchangeBannedError(
                f"IP is banned until {self.rate_limiter.banned_until_ts} — refusing to call the exchange.",
                retry_after_ts=self.rate_limiter.banned_until_ts,
            )

    def _handle_ccxt_exception(self, e: Exception) -> str:
        """Classifies a ccxt exception into 'banned' | 'rate_limited' | 'rejected',
        registers a ban with the tracker if that's what happened, and returns
        the classification so callers can decide what OrderResult.status to use.
        Distinguishing these matters: a generic 'except Exception: return
        rejected' (the previous version of this method) treats an IP ban
        identically to an insufficient-balance rejection, and a caller that
        just moves on to the next signal will keep calling a banned IP,
        extending the ban (Binance's own docs: bans scale from 2 minutes to
        3 days for repeat offenders)."""
        msg = str(e)
        status_code = getattr(e, "http_status", None) or getattr(getattr(e, "args", [None])[0], "http_status", None)
        is_ddos_protection = isinstance(e, getattr(ccxt, "DDoSProtection", ())) if CCXT_AVAILABLE else False
        is_rate_limit_exceeded = isinstance(e, getattr(ccxt, "RateLimitExceeded", ())) if CCXT_AVAILABLE else False

        if "418" in msg or "banned" in msg.lower() or is_ddos_protection:
            self.rate_limiter.register_ban(retry_after_ts=None)  # ccxt error text rarely exposes the exact retryAfter cleanly; use the conservative floor
            return "banned"
        if "429" in msg or is_rate_limit_exceeded:
            logger.warning(f"Rate limited (HTTP 429): {msg}. Back off before retrying — do not immediately resend.")
            return "rate_limited"
        return "rejected"

    def get_symbol_filters(self, symbol: str) -> Dict[str, Any]:
        """Returns ccxt's unified `limits`/`precision` for a symbol (LOT_SIZE
        min qty, MIN_NOTIONAL min cost, amount precision) from the markets
        loaded at connect() time — real, current, per-symbol values from
        `load_markets()`, not the illustrative placeholder constants
        execution/paper_exchange.py uses for local logic-testing."""
        if not self.exchange or symbol not in self.exchange.markets:
            return {}
        m = self.exchange.markets[symbol]
        return {
            "min_qty": (m.get("limits", {}).get("amount", {}) or {}).get("min"),
            "min_notional": (m.get("limits", {}).get("cost", {}) or {}).get("min"),
            "amount_precision": (m.get("precision", {}) or {}).get("amount"),
        }

    def _validate_order_against_filters(self, symbol: str, qty: float, price_estimate: float) -> Optional[str]:
        """Pre-flight check against real exchange filters before spending a
        request (and rate-limit weight) on an order Binance would reject
        anyway. Returns None if OK, or a human-readable rejection reason."""
        filters = self.get_symbol_filters(symbol)
        min_qty = filters.get("min_qty")
        min_notional = filters.get("min_notional")
        if min_qty is not None and qty < min_qty:
            return f"qty {qty} below exchange min_qty {min_qty} for {symbol}"
        if min_notional is not None and price_estimate > 0 and (qty * price_estimate) < min_notional:
            return f"notional {qty * price_estimate:.4f} below exchange min_notional {min_notional} for {symbol}"
        return None

    async def get_ticker(self, symbol: str) -> float:
        """Get mid-price from ticker."""
        self._check_not_banned()
        ticker = await self.exchange.fetch_ticker(symbol)
        self._post_call_bookkeeping()
        bid = ticker.get("bid", 0) or 0
        ask = ticker.get("ask", 0) or 0
        if bid > 0 and ask > 0:
            return (bid + ask) / 2.0
        return ticker.get("last", 0) or 0

    async def get_orderbook(self, symbol: str, depth: int = 5) -> OrderBook:
        self._check_not_banned()
        ob = await self.exchange.fetch_order_book(symbol, limit=depth)
        self._post_call_bookkeeping()
        bids = ob.get("bids", [])
        asks = ob.get("asks", [])
        mid = 0.0
        if bids and asks:
            mid = (bids[0][0] + asks[0][0]) / 2.0
        return OrderBook(
            symbol=symbol,
            bids=bids,
            asks=asks,
            timestamp=time.time(),
            mid_price=mid,
        )

    async def get_balance(self, asset: str) -> float:
        self._check_not_banned()
        balance = await self.exchange.fetch_balance()
        self._post_call_bookkeeping()
        return float(balance.get(asset, {}).get("free", 0))

    async def get_all_balances(self) -> Dict[str, float]:
        self._check_not_banned()
        balance = await self.exchange.fetch_balance()
        self._post_call_bookkeeping()
        result = {}
        for asset, info in balance.get("total", {}).items():
            if isinstance(info, (int, float)) and info > 0:
                result[asset] = float(info)
        return result

    async def place_market_order(self, symbol: str, side: str, qty: float) -> OrderResult:
        """
        Place a market order. side must be 'buy' or 'sell'.

        Returns an OrderResult with fill details. Check `result.status` — one
        of "filled", "rejected" (exchange said no — bad qty, insufficient
        balance, etc.), "invalid" (failed a LOCAL pre-flight filter check,
        never sent to the exchange, no rate-limit weight spent), or "banned"
        (HTTP 418 — the IP is banned; RiskGuardian should treat this as a hard
        halt, not "try the next signal"). This does NOT raise on ordinary
        exchange rejections; it DOES raise `ExchangeBannedError` if already
        inside a known ban window, so a caller with no explicit handling
        fails loudly rather than silently keeps trying a banned IP.
        """
        self._check_not_banned()

        # Pre-flight: check against real exchange filters (loaded at connect()
        # time from load_markets()) before spending a request + rate-limit
        # weight on an order Binance would reject anyway.
        try:
            ticker_price = await self.get_ticker(symbol)
        except ExchangeBannedError:
            raise
        except Exception:
            ticker_price = 0.0  # if we can't get a price estimate, skip the notional check rather than block the order on it

        invalid_reason = self._validate_order_against_filters(symbol, qty, ticker_price)
        if invalid_reason:
            logger.warning(f"Order REJECTED locally (pre-flight, no request sent): {invalid_reason}")
            return OrderResult(
                order_id="", symbol=symbol, side=side, qty=0.0, avg_price=0.0, cost=0.0, fee=0.0,
                timestamp=time.time(), status="invalid", raw={"reason": invalid_reason},
            )

        logger.info(f"Placing market {side} order: {qty} {symbol}")

        try:
            order = await self.exchange.create_market_order(symbol, side, qty)
            self._post_call_bookkeeping()

            fee_cost = 0.0
            if order.get("fee") and order["fee"].get("cost"):
                fee_cost = float(order["fee"]["cost"])

            result = OrderResult(
                order_id=str(order.get("id", "")),
                symbol=symbol,
                side=side,
                qty=float(order.get("filled", qty)),
                avg_price=float(order.get("average", 0)),
                cost=float(order.get("cost", 0)),
                fee=fee_cost,
                timestamp=time.time(),
                status="filled" if order.get("status") == "closed" else order.get("status", "unknown"),
                raw=order,
            )

            logger.info(
                f"Order filled: {result.side} {result.qty} @ {result.avg_price:.2f} "
                f"(cost={result.cost:.4f}, fee={result.fee:.6f})"
            )
            return result

        except Exception as e:
            classification = self._handle_ccxt_exception(e)
            logger.error(f"Order placement failed ({classification}): {e}")
            return OrderResult(
                order_id="",
                symbol=symbol,
                side=side,
                qty=0.0,
                avg_price=0.0,
                cost=0.0,
                fee=0.0,
                timestamp=time.time(),
                status=classification,
                raw={"error": str(e)},
            )

    async def get_open_orders(self, symbol: str) -> List[Dict[str, Any]]:
        self._check_not_banned()
        orders = await self.exchange.fetch_open_orders(symbol)
        self._post_call_bookkeeping()
        return orders

    async def cancel_all_orders(self, symbol: str) -> int:
        self._check_not_banned()
        orders = await self.exchange.fetch_open_orders(symbol)
        self._post_call_bookkeeping()
        count = 0
        for order in orders:
            try:
                await self.exchange.cancel_order(order["id"], symbol)
                self._post_call_bookkeeping()
                count += 1
            except Exception as e:
                logger.warning(f"Failed to cancel order {order['id']}: {e}")
        return count

    async def get_position(self, symbol: str) -> Dict[str, float]:
        """
        Spot doesn't have "positions" — infer from balance.
        For BTC/USDT, position qty = BTC balance.
        """
        base = symbol.split("/")[0] if "/" in symbol else symbol.replace("USDT", "").replace("-", "")
        balance = await self.get_balance(base)
        return {"qty": balance, "entry_price": 0.0}

    async def get_deposit_address(self, asset: str) -> Optional[Dict[str, str]]:
        """
        Real answer to "how do I add funds": this returns YOUR OWN deposit
        address on Binance for `asset` — the dashboard shows this so you can
        send funds from your bank/another wallet/another exchange. There is
        deliberately no code path anywhere in this project that can pull
        funds in or push funds out on its own; funding the account is always
        a manual action you take directly on Binance.
        """
        self._check_not_banned()
        try:
            info = await self.exchange.fetch_deposit_address(asset)
            self._post_call_bookkeeping()
            return {
                "asset": asset,
                "address": info.get("address", ""),
                "network": info.get("network", ""),
                "tag": info.get("tag"),
            }
        except Exception as e:
            logger.error(f"Could not fetch deposit address for {asset}: {e}")
            return None

    async def get_recent_deposits(self, asset: Optional[str] = None, limit: int = 10) -> List[Dict[str, Any]]:
        self._check_not_banned()
        try:
            deposits = await self.exchange.fetch_deposits(code=asset, limit=limit)
            self._post_call_bookkeeping()
            return [
                {
                    "asset": d.get("currency"),
                    "amount": d.get("amount"),
                    "status": d.get("status"),
                    "timestamp": (d.get("timestamp") or 0) / 1000.0,
                    "tx_id": d.get("txid"),
                }
                for d in deposits
            ]
        except Exception as e:
            logger.error(f"Could not fetch deposit history: {e}")
            return []


if __name__ == "__main__":
    logger.info("Exchange adapter module loaded.")
    if CCXT_AVAILABLE:
        logger.info(f"ccxt version: {ccxt.__version__}")
        adapter = BinanceSpotAdapter(testnet=True)
        logger.info(f"Adapter initialized (testnet={adapter.testnet})")
    else:
        logger.error("ccxt not installed. Run: pip install ccxt")
