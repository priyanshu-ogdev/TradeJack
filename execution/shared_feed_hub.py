"""
SharedLiveFeedHub: one real BinanceLiveDepthFeed connection per symbol, fanned
out to every sleeve trading that symbol.

Why this exists: without it, a MultiSleeveOrchestrator running an HFT sleeve
and a position sleeve on the same symbol would naturally end up opening two
separate WebSocket connections (one per LivePaperInferenceServer). That's
wasteful, but the real problem is subtler — two independent connections to the
same Binance stream do not guarantee identical message arrival timing. The two
sleeves' internal order-book reconstructions could drift a message or two out
of sync with each other. That's a fine hair to split for two sleeves acting
independently, but it becomes a real problem the moment anything aggregates
across sleeves (a portfolio-level exposure check, a combined session report,
a decision to net one sleeve's buy against another's sell) — those all
implicitly assume "the book" is one consistent thing at a given instant, and
two sockets don't guarantee that. One connection, fanned out in-process, does.
"""

import time
import logging
import asyncio
import inspect
from typing import Any, Callable, Dict, List

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] (FeedHub) %(message)s")
logger = logging.getLogger("FeedHub")

from execution.binance_live_feed import BinanceLiveDepthFeed, SyntheticReplayFeed, UpdateCallback


class SharedLiveFeedHub:
    """
    Owns exactly one feed connection for `symbol` and dispatches every update
    to every registered subscriber. Subscribers register their own
    `_on_update(kind, payload)` coroutine (e.g. a LivePaperInferenceServer
    constructed with owns_feed=False) — the hub does not know or care what a
    subscriber does with an update, it only guarantees every subscriber sees
    the exact same sequence of messages.
    """

    def __init__(self, symbol: str = "BTC-USDT", use_synthetic_feed: bool = False, depth_levels: int = 20):
        self.symbol = symbol
        self.use_synthetic_feed = use_synthetic_feed
        self._subscribers: List[UpdateCallback] = []
        self._subscriber_names: Dict[int, str] = {}  # id(callback) -> name; bound methods can't hold their own attributes
        self._dispatch_errors: Dict[str, int] = {}

        if use_synthetic_feed:
            logger.warning("SharedLiveFeedHub running SyntheticReplayFeed — NOT real market data.")
            self._feed = SyntheticReplayFeed(symbol=symbol)
        else:
            self._feed = BinanceLiveDepthFeed(symbol=symbol, depth_levels=depth_levels, on_update=self._dispatch)

        self.total_updates_dispatched = 0

    def subscribe(self, callback: UpdateCallback, name: str = "unnamed"):
        """Register a sleeve's _on_update as a callback. Order of registration
        is the order updates are dispatched in — kept sequential (not
        gathered concurrently) so one sleeve's exception can't be blamed on
        message reordering from another sleeve running first."""
        self._subscriber_names[id(callback)] = name  # bound methods disallow arbitrary attribute assignment
        self._subscribers.append(callback)
        logger.info(f"Subscriber '{name}' registered ({len(self._subscribers)} total).")

    async def _dispatch(self, kind: str, payload: Dict[str, Any]):
        self.total_updates_dispatched += 1
        for cb in self._subscribers:
            name = self._subscriber_names.get(id(cb), "unnamed")
            try:
                result = cb(kind, payload)
                if inspect.isawaitable(result):
                    await result
            except Exception as e:
                # One sleeve's bug must not take down every other sleeve sharing
                # this connection — log loudly and keep dispatching to the rest.
                self._dispatch_errors[name] = self._dispatch_errors.get(name, 0) + 1
                logger.error(f"Subscriber '{name}' raised on update #{self.total_updates_dispatched}: {e}")

    async def run(self, duration_sec=None):
        if not self._subscribers:
            logger.warning("SharedLiveFeedHub.run() called with zero subscribers — connecting anyway, but nothing will happen with the data.")
        if self.use_synthetic_feed:
            await self._feed.run(self._dispatch, duration_sec=duration_sec)
        else:
            await self._feed.run(duration_sec=duration_sec)

    def is_stale(self, max_age_sec: float = 2.0) -> bool:
        return self._feed.is_stale(max_age_sec)

    def stop(self):
        self._feed.stop()


if __name__ == "__main__":
    print("Smoke-testing SharedLiveFeedHub fan-out with two fake subscribers (synthetic data, no network)...")

    counts = {"a": 0, "b": 0}

    async def sub_a(kind, payload):
        counts["a"] += 1

    async def sub_b(kind, payload):
        counts["b"] += 1
        if counts["b"] == 3:
            raise RuntimeError("simulated bug in subscriber b")

    async def _main():
        hub = SharedLiveFeedHub(symbol="BTC-USDT", use_synthetic_feed=True)
        hub.subscribe(sub_a, name="a")
        hub.subscribe(sub_b, name="b")
        await hub.run(duration_sec=1.0)
        print(f"Dispatched: {hub.total_updates_dispatched}, sub_a saw: {counts['a']}, sub_b saw: {counts['b']}")
        print(f"Dispatch errors: {hub._dispatch_errors}")
        assert counts["a"] == hub.total_updates_dispatched, "subscriber a should see every update despite b's exception"
        assert hub._dispatch_errors.get("b", 0) >= 1, "b's exception should have been caught and logged, not raised"
        print("SharedLiveFeedHub smoke test passed: one buggy subscriber didn't break the other.")

    asyncio.run(_main())
