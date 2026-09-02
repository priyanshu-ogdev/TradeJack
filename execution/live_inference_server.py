"""
Live Inference Server — Production inference loop with frozen model weights.

Reads live market data → feeds to frozen SB3 model → routes through RiskGuardian →
places real orders on exchange.

The model weights are FROZEN. No training happens here.
Model swaps happen atomically via file replacement + reload.

Modes:
  paper:   PaperExchangeAdapter (no real orders)
  testnet: BinanceSpotAdapter(testnet=True)
  live:    BinanceSpotAdapter(testnet=False) — requires testnet evidence

Usage:
    server = LiveInferenceServer.from_config(deploy_config)
    await server.run_forever()
"""

import os
import sys
import time
import asyncio
import logging
import numpy as np
import sqlite3
from typing import Dict, Any, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] (InferenceServer) %(message)s")
logger = logging.getLogger("InferenceServer")

try:
    from stable_baselines3 import PPO, SAC, DQN
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False

from execution.exchange_adapter import ExchangeAdapter, BinanceSpotAdapter
from execution.paper_exchange import PaperExchangeAdapter
from execution.risk_guardian import RiskGuardian
from execution.position_throttle import PositionThrottle


class LiveInferenceServer:
    """
    Production inference loop.

    Architecture:
      Live Market Data → Observation Builder → Frozen Model.predict()
        → Position Delta → RiskGuardian.execute_safe_order() → Exchange

    The server runs in an async loop, processing one tick per cycle.
    Each cycle: fetch price → build obs → predict → check risk → execute → record.
    """

    def __init__(
        self,
        exchange: ExchangeAdapter,
        risk_guardian: RiskGuardian,
        position_throttle: PositionThrottle,
        model_path: str,
        model_name: str = "PPO-DilatedCNN",
        symbol: str = "BTC/USDT",
        tick_interval_sec: float = 60.0,
        seq_len: int = 64,
        state_dir: str = "d:/TradeJack/state/deployed",
    ):
        self.exchange = exchange
        self.risk_guardian = risk_guardian
        self.throttle = position_throttle
        self.model_path = model_path
        self.model_name = model_name
        self.symbol = symbol
        self.tick_interval_sec = tick_interval_sec
        self.seq_len = seq_len
        self.state_dir = state_dir

        # Model state
        self.model = None
        self._load_model()

        # Observation history for building sequences
        self.price_history: list = []
        self.volume_history: list = []

        # Portfolio tracking
        self.current_equity = 0.0
        self.prev_equity = 0.0
        self.position_qty = 0.0
        self.tick_count = 0
        self.is_running = False

        # Telemetry DB
        self.telemetry_db_path = os.path.join(state_dir, "inference_telemetry.sqlite")
        os.makedirs(state_dir, exist_ok=True)
        self._init_telemetry_db()

    def _load_model(self):
        """Load frozen SB3 model from checkpoint."""
        if not SB3_AVAILABLE:
            logger.error("SB3 not available. Cannot load model.")
            return

        if not os.path.exists(self.model_path) and not os.path.exists(self.model_path + ".zip"):
            logger.warning(f"Model checkpoint not found at {self.model_path}. Using random policy.")
            return

        try:
            # Detect algorithm from model_name
            algo_map = {"PPO": PPO, "SAC": SAC, "DQN": DQN}
            algo_prefix = self.model_name.split("-")[0] if "-" in self.model_name else "PPO"
            algo_cls = algo_map.get(algo_prefix, PPO)

            self.model = algo_cls.load(self.model_path, device="cpu")
            logger.info(f"Loaded frozen model: {self.model_name} from {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.model = None

    def hot_swap_model(self, new_model_path: str):
        """Atomically swap the inference model. Called after promotion."""
        logger.info(f"Hot-swapping model: {self.model_path} → {new_model_path}")
        old_path = self.model_path
        self.model_path = new_model_path
        try:
            self._load_model()
            logger.info("Model hot-swap complete.")
        except Exception as e:
            logger.error(f"Hot-swap failed, reverting: {e}")
            self.model_path = old_path
            self._load_model()

    def _init_telemetry_db(self):
        """Initialize telemetry database for recording inference decisions."""
        conn = sqlite3.connect(self.telemetry_db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS inference_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL,
                tick INTEGER,
                price REAL,
                action REAL,
                position_qty REAL,
                equity REAL,
                throttle_fraction REAL,
                order_placed INTEGER,
                order_side TEXT,
                order_qty REAL
            )
        """)
        conn.commit()
        conn.close()

    def _log_telemetry(self, price, action, order_placed, order_side, order_qty):
        try:
            conn = sqlite3.connect(self.telemetry_db_path)
            conn.execute(
                "INSERT INTO inference_log "
                "(timestamp, tick, price, action, position_qty, equity, throttle_fraction, order_placed, order_side, order_qty) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (time.time(), self.tick_count, price, action, self.position_qty,
                 self.current_equity, self.throttle.get_throttled_fraction(),
                 int(order_placed), order_side or "", order_qty)
            )
            conn.commit()
            conn.close()
        except Exception as e:
            logger.debug(f"Telemetry log error: {e}")

    def _build_observation(self, current_price: float) -> Dict[str, np.ndarray]:
        """
        Build a model-compatible observation from price history.

        Returns dict with:
          lob_sequence: (seq_len, 5) — [close, volume, ofi, vpin, kyles_lambda]
          portfolio_state: (4,) — [cash_norm, position_qty, equity_norm, max_drawdown]
        """
        # Pad history if too short
        while len(self.price_history) < self.seq_len:
            self.price_history.insert(0, current_price)
        while len(self.volume_history) < self.seq_len:
            self.volume_history.insert(0, 0.0)

        prices = np.array(self.price_history[-self.seq_len:], dtype=np.float32)
        volumes = np.array(self.volume_history[-self.seq_len:], dtype=np.float32)

        # Compute derived features (simplified for live — full version uses feature_engineering)
        returns = np.diff(prices, prepend=prices[0])
        ofi = np.sign(returns) * volumes  # Simplified OFI
        vpin = np.abs(ofi) / (volumes + 1e-8)  # Simplified VPIN
        kyles_lambda = np.abs(returns) / (volumes + 1e-8)  # Simplified Kyle's Lambda

        # Normalize
        price_mean, price_std = np.mean(prices), np.std(prices) + 1e-8
        norm_prices = (prices - price_mean) / price_std

        lob_seq = np.stack([norm_prices, volumes, ofi, vpin, kyles_lambda], axis=1).astype(np.float32)

        # Clip to observation space bounds
        lob_seq = np.clip(lob_seq, -100.0, 100.0)

        equity = self.current_equity if self.current_equity > 0 else 100.0
        dd = self.risk_guardian.state.peak_equity - self.current_equity
        dd_pct = dd / max(self.risk_guardian.state.peak_equity, 1e-8)

        port_state = np.array([
            self.current_equity / max(equity, 1e-8),  # cash_norm
            self.position_qty,
            self.current_equity / max(equity, 1e-8),  # equity_norm
            dd_pct,
        ], dtype=np.float32)
        port_state = np.clip(port_state, -100.0, 100.0)

        return {"lob_sequence": lob_seq, "portfolio_state": port_state}

    def _predict_action(self, obs: Dict[str, np.ndarray]) -> float:
        """Run frozen model inference. Returns target position fraction [-1, 1]."""
        if self.model is None:
            return 0.0  # No model loaded — hold flat

        try:
            action, _ = self.model.predict(obs, deterministic=True)
            if isinstance(action, np.ndarray):
                return float(np.clip(action[0], -1.0, 1.0))
            return float(np.clip(action, -1.0, 1.0))
        except Exception as e:
            logger.error(f"Model predict error: {e}")
            return 0.0

    async def _execute_tick(self):
        """Single inference tick: observe → think → act."""
        # 1. Get current price
        current_price = await self.exchange.get_ticker(self.symbol)
        if current_price <= 0:
            logger.warning("Invalid price received. Skipping tick.")
            return

        self.price_history.append(current_price)

        # 2. Update equity
        pos = await self.exchange.get_position(self.symbol)
        self.position_qty = pos.get("qty", 0.0)
        quote_balance = await self.exchange.get_balance("USDT")
        self.prev_equity = self.current_equity
        self.current_equity = quote_balance + self.position_qty * current_price

        self.risk_guardian.update_equity(self.current_equity)

        # Update throttle
        if self.prev_equity > 0:
            self.throttle.record_equity_change(self.prev_equity, self.current_equity)

        # 3. Build observation and predict
        obs = self._build_observation(current_price)
        raw_action = self._predict_action(obs)

        # 4. Apply position throttle
        throttled_fraction = self.throttle.get_throttled_fraction()
        scaled_action = raw_action * throttled_fraction

        # 5. Compute position delta
        target_value = scaled_action * self.current_equity
        target_qty = target_value / current_price if current_price > 0 else 0.0
        delta_qty = target_qty - self.position_qty

        # 6. Execute through risk guardian if delta is meaningful
        order_placed = False
        order_side = None
        order_qty = 0.0
        min_notional = 5.0  # Binance minimum notional

        if abs(delta_qty * current_price) > min_notional:
            side = "buy" if delta_qty > 0 else "sell"
            qty = abs(delta_qty)

            result = await self.risk_guardian.execute_safe_order(side, qty, current_price)
            if result and result.status == "filled":
                order_placed = True
                order_side = side
                order_qty = result.qty

        # 7. Reconcile positions periodically
        await self.risk_guardian.reconcile_positions()

        # 8. Log telemetry
        self._log_telemetry(current_price, raw_action, order_placed, order_side, order_qty)
        self.tick_count += 1

        if self.tick_count % 60 == 0:  # Log summary every 60 ticks
            logger.info(
                f"[Tick {self.tick_count}] price={current_price:.2f} "
                f"action={raw_action:.3f} scaled={scaled_action:.3f} "
                f"equity=${self.current_equity:.2f} "
                f"throttle={throttled_fraction:.3f} "
                f"pos={self.position_qty:.6f}"
            )

    async def run_forever(self, max_ticks: Optional[int] = None):
        """
        Main inference loop. Runs until halted, max_ticks reached, or interrupted.
        """
        logger.info(
            f"LiveInferenceServer starting: symbol={self.symbol}, "
            f"tick_interval={self.tick_interval_sec}s, model={self.model_name}"
        )

        await self.exchange.connect()
        self.is_running = True

        try:
            while self.is_running:
                if self.risk_guardian.state.is_halted:
                    logger.warning("Trading halted. Waiting for release...")
                    await asyncio.sleep(30)
                    continue

                try:
                    await self._execute_tick()
                except Exception as e:
                    logger.error(f"Tick error: {e}")

                    # If exchange disconnected, try emergency flatten
                    if not self.exchange.is_connected():
                        logger.critical("Exchange connection lost!")
                        await self.risk_guardian.emergency_flatten("Connection lost")

                if max_ticks and self.tick_count >= max_ticks:
                    logger.info(f"Max ticks ({max_ticks}) reached. Stopping.")
                    break

                await asyncio.sleep(self.tick_interval_sec)

        except asyncio.CancelledError:
            logger.info("Inference server cancelled.")
        finally:
            self.is_running = False
            await self.exchange.close()
            logger.info(f"Inference server stopped after {self.tick_count} ticks.")

    def stop(self):
        """Signal the server to stop."""
        self.is_running = False

    def get_status(self) -> Dict[str, Any]:
        """Current server status for dashboard."""
        return {
            "running": self.is_running,
            "tick_count": self.tick_count,
            "model": self.model_name,
            "model_loaded": self.model is not None,
            "equity": round(self.current_equity, 2),
            "position_qty": self.position_qty,
            "throttle": self.throttle.get_status(),
            "risk": self.risk_guardian.get_risk_summary(),
        }

    @staticmethod
    def from_config(config) -> "LiveInferenceServer":
        """Create a LiveInferenceServer from DeploymentConfig."""
        # Select exchange adapter
        if config.exchange_mode == "paper":
            exchange = PaperExchangeAdapter(initial_balance_usdt=config.starting_capital)
        elif config.exchange_mode in ("testnet", "live"):
            testnet = (config.exchange_mode == "testnet")
            exchange = BinanceSpotAdapter(
                testnet=testnet,
                api_key_env=config.api_key_env_var,
                api_secret_env=config.api_secret_env_var,
            )
        else:
            raise ValueError(f"Unknown exchange_mode: {config.exchange_mode}")

        # Build risk guardian
        symbol = config.symbol.replace("-", "/")
        audit_path = os.path.join("state", "deployed", "risk_audit.sqlite")
        risk_guardian = RiskGuardian(
            exchange=exchange,
            symbol=symbol,
            max_position_fraction=config.max_position_fraction,
            max_daily_loss_pct=config.max_daily_loss_pct,
            max_drawdown_halt=config.max_drawdown_halt,
            min_hold_ticks=config.min_hold_ticks,
            max_orders_per_minute=config.max_orders_per_minute,
            connection_loss_flatten_sec=config.connection_loss_flatten_sec,
            order_reconciliation_interval_sec=config.order_reconciliation_interval_sec,
            starting_equity=config.starting_capital,
            audit_db_path=audit_path,
        )

        # Build position throttle
        throttle = PositionThrottle(
            base_fraction=config.max_position_fraction,
            sortino_full=config.throttle_sortino_full,
            sortino_zero=config.throttle_sortino_zero,
            min_fraction=config.throttle_min_fraction,
        )

        return LiveInferenceServer(
            exchange=exchange,
            risk_guardian=risk_guardian,
            position_throttle=throttle,
            model_path=config.frozen_model_path,
            model_name=config.model_name,
            symbol=symbol,
            state_dir=os.path.join("state", "deployed"),
        )


if __name__ == "__main__":
    from scripts.deploy_config import DEPLOY_CONFIG

    async def test_paper():
        server = LiveInferenceServer.from_config(DEPLOY_CONFIG)
        logger.info(f"Server status: {server.get_status()}")
        await server.run_forever(max_ticks=10)
        logger.info(f"Final status: {server.get_status()}")

    asyncio.run(test_paper())
