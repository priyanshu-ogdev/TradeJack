"""
Paper Exchange — Simulated exchange using LOB physics for paper trading.

Same API surface as BinanceSpotAdapter but fills orders using the LOB
physics engine's slippage model. Tracks a virtual balance. Perfect for
full pipeline validation before touching testnet/mainnet.

Usage:
    paper = PaperExchangeAdapter(initial_balance_usdt=100.0)
    await paper.connect()  # No-op, always succeeds
    result = await paper.place_market_order("BTC/USDT", "buy", 0.001)
    balance = await paper.get_balance("USDT")
"""

import time
import logging
import asyncio
from typing import List, Dict, Any, Optional

from execution.exchange_adapter import ExchangeAdapter, OrderResult, OrderBook

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] (PaperExchange) %(message)s")
logger = logging.getLogger("PaperExchange")


class PaperExchangeAdapter(ExchangeAdapter):
    """
    Simulated exchange for paper trading.

    Uses TradeJackLOBEnv.compute_friction_fill_price() for realistic
    slippage modeling. No real orders are placed.

    Tracks:
      - Virtual balances per asset
      - Simulated order history
      - Fill prices with Kyle's Lambda slippage
    """

    def __init__(
        self,
        initial_balance_usdt: float = 100.0,
        taker_fee_bps: float = 10.0,
        simulated_kyles_lambda: float = 0.001,
    ):
        self.balances: Dict[str, float] = {"USDT": initial_balance_usdt}
        self.taker_fee_rate = taker_fee_bps / 10_000.0
        self.kyles_lambda = simulated_kyles_lambda
        self._connected = True
        self.order_history: List[OrderResult] = []
        self._order_counter = 0

        # Simulated price feed (updated on each get_ticker/get_orderbook call)
        self._current_prices: Dict[str, float] = {
            "BTC/USDT": 60000.0,
            "ETH/USDT": 3000.0,
            "SOL/USDT": 150.0,
        }

    async def connect(self) -> None:
        self._connected = True
        logger.info(f"Paper exchange connected. Balances: {self.balances}")

    async def close(self) -> None:
        self._connected = False
        logger.info("Paper exchange closed.")

    def is_connected(self) -> bool:
        return self._connected

    def set_price(self, symbol: str, price: float):
        """External price feed injection (e.g., from live WebSocket data)."""
        self._current_prices[symbol] = price

    async def get_ticker(self, symbol: str) -> float:
        return self._current_prices.get(symbol, 0.0)

    async def get_orderbook(self, symbol: str, depth: int = 5) -> OrderBook:
        mid = self._current_prices.get(symbol, 0.0)
        if mid <= 0:
            return OrderBook(symbol=symbol, bids=[], asks=[], timestamp=time.time(), mid_price=0.0)

        spread = mid * 0.0001  # 1 bps spread
        bids = [[mid - spread * (i + 1), 1.0 + i * 0.5] for i in range(depth)]
        asks = [[mid + spread * (i + 1), 1.0 + i * 0.5] for i in range(depth)]

        return OrderBook(
            symbol=symbol, bids=bids, asks=asks,
            timestamp=time.time(), mid_price=mid,
        )

    async def get_balance(self, asset: str) -> float:
        return self.balances.get(asset, 0.0)

    async def get_all_balances(self) -> Dict[str, float]:
        return {k: v for k, v in self.balances.items() if v > 0}

    def _compute_fill_price(self, symbol: str, side: str, qty: float) -> float:
        """
        Compute fill price with Kyle's Lambda slippage.

        Slippage = lambda * abs(qty)
        Buy: fill_price = mid + slippage
        Sell: fill_price = mid - slippage
        """
        mid = self._current_prices.get(symbol, 0.0)
        if mid <= 0:
            return 0.0

        slippage = self.kyles_lambda * abs(qty)
        if side == "buy":
            return mid + slippage * mid  # Proportional slippage
        else:
            return mid - slippage * mid

    async def place_market_order(self, symbol: str, side: str, qty: float) -> OrderResult:
        """
        Simulate a market order with slippage and fees.

        Updates virtual balances accordingly.
        """
        self._order_counter += 1
        order_id = f"PAPER-{self._order_counter:06d}"

        fill_price = self._compute_fill_price(symbol, side, qty)
        if fill_price <= 0:
            return OrderResult(
                order_id=order_id, symbol=symbol, side=side, qty=0.0,
                avg_price=0.0, cost=0.0, fee=0.0, timestamp=time.time(),
                status="rejected", raw={"error": f"No price for {symbol}"},
            )

        notional = qty * fill_price
        fee = notional * self.taker_fee_rate

        # Parse base/quote from symbol
        parts = symbol.replace("-", "/").split("/")
        base = parts[0] if len(parts) >= 2 else symbol
        quote = parts[1] if len(parts) >= 2 else "USDT"

        if side == "buy":
            # Check sufficient quote balance
            total_cost = notional + fee
            if self.balances.get(quote, 0.0) < total_cost:
                return OrderResult(
                    order_id=order_id, symbol=symbol, side=side, qty=0.0,
                    avg_price=fill_price, cost=0.0, fee=0.0, timestamp=time.time(),
                    status="rejected", raw={"error": "Insufficient balance"},
                )
            self.balances[quote] = self.balances.get(quote, 0.0) - total_cost
            self.balances[base] = self.balances.get(base, 0.0) + qty

        elif side == "sell":
            # Check sufficient base balance
            if self.balances.get(base, 0.0) < qty:
                return OrderResult(
                    order_id=order_id, symbol=symbol, side=side, qty=0.0,
                    avg_price=fill_price, cost=0.0, fee=0.0, timestamp=time.time(),
                    status="rejected", raw={"error": "Insufficient balance"},
                )
            self.balances[base] = self.balances.get(base, 0.0) - qty
            self.balances[quote] = self.balances.get(quote, 0.0) + notional - fee

        result = OrderResult(
            order_id=order_id, symbol=symbol, side=side, qty=qty,
            avg_price=fill_price, cost=notional, fee=fee,
            timestamp=time.time(), status="filled",
        )
        self.order_history.append(result)

        logger.info(
            f"[PAPER] {side.upper()} {qty:.6f} {base} @ {fill_price:.2f} "
            f"(cost={notional:.4f} {quote}, fee={fee:.6f})"
        )
        return result

    async def get_open_orders(self, symbol: str) -> List[Dict[str, Any]]:
        return []  # Paper exchange fills immediately

    async def cancel_all_orders(self, symbol: str) -> int:
        return 0  # Nothing to cancel

    async def get_position(self, symbol: str) -> Dict[str, float]:
        parts = symbol.replace("-", "/").split("/")
        base = parts[0] if len(parts) >= 2 else symbol
        qty = self.balances.get(base, 0.0)
        return {"qty": qty, "entry_price": 0.0}

    def get_equity(self, base_symbol: str = "BTC/USDT") -> float:
        """Total equity in quote currency (USDT)."""
        total = self.balances.get("USDT", 0.0)
        for asset, qty in self.balances.items():
            if asset == "USDT" or qty <= 0:
                continue
            sym = f"{asset}/USDT"
            price = self._current_prices.get(sym, 0.0)
            total += qty * price
        return total

    def get_trade_summary(self) -> Dict[str, Any]:
        """Summary of all paper trades."""
        buys = [o for o in self.order_history if o.side == "buy" and o.status == "filled"]
        sells = [o for o in self.order_history if o.side == "sell" and o.status == "filled"]
        total_fees = sum(o.fee for o in self.order_history if o.status == "filled")
        return {
            "total_trades": len(self.order_history),
            "filled_buys": len(buys),
            "filled_sells": len(sells),
            "total_fees": round(total_fees, 6),
            "final_equity": round(self.get_equity(), 4),
            "balances": {k: round(v, 8) for k, v in self.balances.items() if v > 0},
        }


if __name__ == "__main__":
    async def test_paper():
        paper = PaperExchangeAdapter(initial_balance_usdt=100.0)
        await paper.connect()

        logger.info(f"Initial USDT: {await paper.get_balance('USDT')}")
        logger.info(f"BTC price: {await paper.get_ticker('BTC/USDT')}")

        # Buy some BTC
        result = await paper.place_market_order("BTC/USDT", "buy", 0.001)
        logger.info(f"Buy result: {result.status}, filled={result.qty}")

        # Sell it back
        result = await paper.place_market_order("BTC/USDT", "sell", 0.001)
        logger.info(f"Sell result: {result.status}, filled={result.qty}")

        logger.info(f"Trade summary: {paper.get_trade_summary()}")
        await paper.close()

    asyncio.run(test_paper())
