"""
Exchange Adapter — Abstract exchange interface + Binance Spot implementation.

Uses ccxt for Binance API interaction. Supports testnet and mainnet via config.
Secrets loaded from environment variables (never stored in code).

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
from dataclasses import dataclass
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
    status: str         # "filled", "partial", "rejected"
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

    @abstractmethod
    def is_connected(self) -> bool:
        """Check if exchange connection is healthy."""


class BinanceSpotAdapter(ExchangeAdapter):
    """
    Binance Spot exchange adapter via ccxt.

    Supports both testnet (testnet.binance.vision) and mainnet (api.binance.com).
    API keys loaded from environment variables specified in DeploymentConfig.

    Rate limiting is handled by ccxt internally.
    """

    # Binance Spot testnet URLs
    TESTNET_URL = "https://testnet.binance.vision"
    TESTNET_WS = "wss://testnet.binance.vision/ws"

    def __init__(
        self,
        testnet: bool = True,
        api_key_env: str = "BINANCE_API_KEY",
        api_secret_env: str = "BINANCE_API_SECRET",
    ):
        if not CCXT_AVAILABLE:
            raise ImportError("ccxt required for BinanceSpotAdapter. pip install ccxt")

        self.testnet = testnet
        self.api_key = os.environ.get(api_key_env, "")
        self.api_secret = os.environ.get(api_secret_env, "")
        self.exchange: Optional[ccxt_async.binance] = None
        self._connected = False
        self._last_heartbeat = 0.0

        if not self.api_key or not self.api_secret:
            logger.warning(
                f"Exchange API keys not found in env vars ({api_key_env}, {api_secret_env}). "
                "Set them in .env file. Orders will fail."
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

    async def get_ticker(self, symbol: str) -> float:
        """Get mid-price from ticker."""
        ticker = await self.exchange.fetch_ticker(symbol)
        self._last_heartbeat = time.time()
        bid = ticker.get("bid", 0) or 0
        ask = ticker.get("ask", 0) or 0
        if bid > 0 and ask > 0:
            return (bid + ask) / 2.0
        return ticker.get("last", 0) or 0

    async def get_orderbook(self, symbol: str, depth: int = 5) -> OrderBook:
        ob = await self.exchange.fetch_order_book(symbol, limit=depth)
        self._last_heartbeat = time.time()
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
        balance = await self.exchange.fetch_balance()
        self._last_heartbeat = time.time()
        return float(balance.get(asset, {}).get("free", 0))

    async def get_all_balances(self) -> Dict[str, float]:
        balance = await self.exchange.fetch_balance()
        self._last_heartbeat = time.time()
        result = {}
        for asset, info in balance.get("total", {}).items():
            if isinstance(info, (int, float)) and info > 0:
                result[asset] = float(info)
        return result

    async def place_market_order(self, symbol: str, side: str, qty: float) -> OrderResult:
        """
        Place a market order. side must be 'buy' or 'sell'.

        Returns an OrderResult with fill details.
        Raises on exchange errors — caller (RiskGuardian) handles retries.
        """
        logger.info(f"Placing market {side} order: {qty} {symbol}")

        try:
            order = await self.exchange.create_market_order(symbol, side, qty)
            self._last_heartbeat = time.time()

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
            logger.error(f"Order placement failed: {e}")
            return OrderResult(
                order_id="",
                symbol=symbol,
                side=side,
                qty=0.0,
                avg_price=0.0,
                cost=0.0,
                fee=0.0,
                timestamp=time.time(),
                status="rejected",
                raw={"error": str(e)},
            )

    async def get_open_orders(self, symbol: str) -> List[Dict[str, Any]]:
        orders = await self.exchange.fetch_open_orders(symbol)
        self._last_heartbeat = time.time()
        return orders

    async def cancel_all_orders(self, symbol: str) -> int:
        orders = await self.exchange.fetch_open_orders(symbol)
        self._last_heartbeat = time.time()
        count = 0
        for order in orders:
            try:
                await self.exchange.cancel_order(order["id"], symbol)
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


if __name__ == "__main__":
    logger.info("Exchange adapter module loaded.")
    if CCXT_AVAILABLE:
        logger.info(f"ccxt version: {ccxt.__version__}")
        adapter = BinanceSpotAdapter(testnet=True)
        logger.info(f"Adapter initialized (testnet={adapter.testnet})")
    else:
        logger.error("ccxt not installed. Run: pip install ccxt")
