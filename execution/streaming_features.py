"""
Streaming feature computation for the live-paper-trading loop.

IMPORTANT — TRAIN/SERVE SKEW WARNING, read before trusting any live-paper result:
Historical training data is produced by data_forge/feature_engineering.py
(`TradeFlowPhysics`), which computes OFI / VPIN / Kyle's-lambda in **daily batch
jobs** over volume-bucketed bars, carrying a "previous remainder" state file
across day boundaries so bucket boundaries are continuous across the whole
history. That is a file-based batch pipeline — it cannot be called tick-by-tick
on a live socket without material rework, and this upgrade was told not to touch
data_forge/, so it hasn't been.

What's here instead is an **independent, incremental approximation** of the same
three feature families, computed directly from the live order book and trade
stream:
  - OFI: the standard best-level order-flow-imbalance definition (Cont, Kukanov &
    Stoikov, 2014), EWMA-smoothed.
  - VPIN-style toxicity: volume-bucketed buy/sell imbalance, buckets classified
    by Binance aggTrade's `is_buyer_maker` flag (a real aggressor-side label, not
    a tick-rule guess).
  - Kyle's lambda: rolling OLS slope of signed trade volume against the resulting
    price change (the actual Kyle's-lambda estimator, not the closed-form
    physics/lob_env.py linear friction formula, which assumes lambda is already a
    known/precomputed feature rather than estimating it live).

This is close in spirit but NOT numerically identical to what any model was
trained on — different bucket boundaries, different smoothing, different
warm-up behavior. A frozen model fed these features is being asked to
generalize across that gap. Treat any live-paper P&L number as informative about
"does this policy still do something sensible on real data", not as a
production-equivalent replay. Closing this gap properly means porting
TradeFlowPhysics's bucket logic to a streaming form — worth doing explicitly,
later, as its own reviewed change to data_forge.
"""

import math
from collections import deque
from typing import Any, Deque, Dict, Optional

import numpy as np


class StreamingFeatureEngine:
    """
    Consumes live depth snapshots (from binance_live_feed / SyntheticReplayFeed)
    and trade prints, and emits the 5-channel feature vector
    [mid_price, rolling_volume, ofi, vpin, kyles_lambda] that mirrors the
    (close_price, volume, ofi, vpin_50, kyles_lambda) schema physics/lob_env.py
    uses, so a model trained on that schema has a same-shaped input to run on.
    """

    def __init__(self, volume_bucket_size: float = 10.0, ofi_ewma_alpha: float = 0.1, vol_window: int = 50, vpin_bucket_history: int = 50):
        self.volume_bucket_size = volume_bucket_size
        self.ofi_alpha = ofi_ewma_alpha
        self.vol_window = vol_window

        self.prev_best_bid_px: Optional[float] = None
        self.prev_best_bid_sz: Optional[float] = None
        self.prev_best_ask_px: Optional[float] = None
        self.prev_best_ask_sz: Optional[float] = None
        self.ofi_ewma = 0.0

        self.last_mid: Optional[float] = None
        self.rolling_volume_window: Deque[float] = deque(maxlen=vol_window)
        self.signed_volume_history: Deque[float] = deque(maxlen=vol_window)
        self.price_change_history: Deque[float] = deque(maxlen=vol_window)

        self.bucket_buy_vol = 0.0
        self.bucket_sell_vol = 0.0
        self.vpin_buckets: Deque[float] = deque(maxlen=vpin_bucket_history)

        self.n_depth_updates = 0
        self.n_trades = 0

    def on_depth(self, snap: Dict[str, Any]) -> Optional[np.ndarray]:
        bid_px, bid_sz = float(snap.get("bid_px_0", 0.0)), float(snap.get("bid_sz_0", 0.0))
        ask_px, ask_sz = float(snap.get("ask_px_0", 0.0)), float(snap.get("ask_sz_0", 0.0))
        if bid_px <= 0 or ask_px <= 0 or ask_px <= bid_px:
            return None

        mid = (bid_px + ask_px) / 2.0

        ofi_step = 0.0
        if self.prev_best_bid_px is not None:
            if bid_px > self.prev_best_bid_px:
                ofi_step += bid_sz
            elif bid_px == self.prev_best_bid_px:
                ofi_step += (bid_sz - self.prev_best_bid_sz)
            else:
                ofi_step -= self.prev_best_bid_sz

            if ask_px < self.prev_best_ask_px:
                ofi_step -= ask_sz
            elif ask_px == self.prev_best_ask_px:
                ofi_step -= (ask_sz - self.prev_best_ask_sz)
            else:
                ofi_step += self.prev_best_ask_sz
        self.ofi_ewma = (1 - self.ofi_alpha) * self.ofi_ewma + self.ofi_alpha * ofi_step

        self.prev_best_bid_px, self.prev_best_bid_sz = bid_px, bid_sz
        self.prev_best_ask_px, self.prev_best_ask_sz = ask_px, ask_sz
        self.last_mid = mid
        self.n_depth_updates += 1

        volume_feature = float(np.sum(self.rolling_volume_window)) if self.rolling_volume_window else 0.0
        kyles_lambda = self._estimate_kyles_lambda()
        vpin = float(np.mean(self.vpin_buckets)) if self.vpin_buckets else 0.0

        return np.array([mid, volume_feature, self.ofi_ewma, vpin, kyles_lambda], dtype=np.float32)

    def on_trade(self, trade: Dict[str, Any]):
        qty = float(trade["qty"])
        price = float(trade["price"])
        # Binance aggTrade: is_buyer_maker=True means the resting order was a buy,
        # so the aggressor (the side that moved the market) was the SELLER.
        is_sell_aggressor = bool(trade["is_buyer_maker"])
        signed_qty = -qty if is_sell_aggressor else qty

        self.rolling_volume_window.append(qty)
        self.n_trades += 1

        if self.last_mid is not None:
            self.price_change_history.append(price - self.last_mid)
            self.signed_volume_history.append(signed_qty)

        if signed_qty > 0:
            self.bucket_buy_vol += qty
        else:
            self.bucket_sell_vol += qty  # qty is already positive; sign tracked separately

        bucket_total = self.bucket_buy_vol + self.bucket_sell_vol
        if bucket_total >= self.volume_bucket_size:
            imbalance = abs(self.bucket_buy_vol - self.bucket_sell_vol)
            self.vpin_buckets.append(imbalance / bucket_total if bucket_total > 0 else 0.0)
            self.bucket_buy_vol = 0.0
            self.bucket_sell_vol = 0.0

    def _estimate_kyles_lambda(self) -> float:
        """Rolling OLS slope: price change ~ signed trade volume. This is the
        actual Kyle's-lambda estimator (price impact per unit signed volume),
        distinct from physics/lob_env.py's `compute_friction_fill_price`, which
        treats kyles_lambda as an already-known input rather than estimating it."""
        if len(self.signed_volume_history) < 10:
            return 0.0
        x = np.asarray(self.signed_volume_history, dtype=np.float64)
        y = np.asarray(self.price_change_history, dtype=np.float64)
        var_x = float(np.var(x))
        if var_x < 1e-12:
            return 0.0
        cov_xy = float(np.mean((x - x.mean()) * (y - y.mean())))
        return cov_xy / var_x


if __name__ == "__main__":
    import asyncio
    from execution.binance_live_feed import SyntheticReplayFeed  # synthetic only — see that module's warning

    print("Smoke-testing StreamingFeatureEngine against SyntheticReplayFeed (not real data)...")
    engine = StreamingFeatureEngine(volume_bucket_size=1.0)

    def _on_update(kind, payload):
        if kind == "depth":
            feat = engine.on_depth(payload)
            if feat is not None and engine.n_depth_updates % 5 == 0:
                print(f"depth#{engine.n_depth_updates} feat={feat}")
        else:
            engine.on_trade(payload)

    feed = SyntheticReplayFeed(seed=1)
    asyncio.run(feed.run(_on_update, duration_sec=2.0, tick_hz=20.0))
    print(f"Total depth updates: {engine.n_depth_updates}, trades: {engine.n_trades}")
