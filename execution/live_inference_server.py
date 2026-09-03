"""
Live-paper inference server: wires binance_live_feed -> streaming_features ->
a frozen (never-trained-during-this-loop) model -> risk_guardian ->
paper_exchange into one continuous loop, and logs every decision — approved,
rejected, and filled — for review afterward via session_report.py.

This is the Phase-4-per-the-plan-discussion piece: real Binance market data,
synthetic wallet, no third-party demo broker. It answers "if these frozen
weights were released right now, what would happen" as honestly as an
in-process simulation can, which is why every friction (latency, fees, visible
depth, staleness) is modeled rather than assumed away.

Nothing here trains the model or places a real order. Model weights are loaded
once at startup and never updated (mirrors scripts/deploy_config.py's frozen-
deployment philosophy) — this is deliberately a *validation* stage, not the
self-RL training loop (that's swarm/rl_trainer.py + training/continuous_trainer.py,
separate work).
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import time
import json
import sqlite3
import logging
import argparse
import asyncio
from collections import deque
from typing import Any, Deque, Dict, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] (LiveInference) %(message)s")
logger = logging.getLogger("LiveInference")

import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from stable_baselines3 import PPO, SAC, DQN
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False

from swarm.model_registry import REGISTRY
from swarm.baselines import MomentumBaseline
from scripts.deploy_config import DEPLOY_CONFIG
from execution.binance_live_feed import BinanceLiveDepthFeed, SyntheticReplayFeed
from execution.streaming_features import StreamingFeatureEngine
from execution.paper_exchange import PaperExchange
from execution.risk_guardian import RiskGuardian, RiskLimits


def load_frozen_model(weights_path: str, model_name: str = "PPO-DilatedCNN", input_dim: int = 5) -> Any:
    """
    Loads frozen weights for live paper inference.
    Supports:
      1. SB3 checkpoints (.zip) loaded via PPO/SAC/DQN
      2. PyTorch state_dict (.pt) checkpoints
      3. Rule-based baselines (Momentum-Baseline, BuyAndHold-Baseline)
      4. Safe fallback if promoted checkpoint is not yet generated
    """
    card = REGISTRY.get_model_card(model_name)
    algo_class = card.algo_class if card else "PPO"

    # 1. Attempt SB3 checkpoint load
    algo_map = {"PPO": PPO, "SAC": SAC, "DQN": DQN} if SB3_AVAILABLE else {}
    target_cls = algo_map.get(algo_class)

    candidates = [weights_path]
    if weights_path and not weights_path.endswith(".zip"):
        candidates.append(weights_path + ".zip")

    for path in candidates:
        if path and os.path.exists(path):
            if SB3_AVAILABLE and target_cls is not None:
                try:
                    model = target_cls.load(path, device="cpu")
                    logger.info(f"Loaded frozen SB3 model from {path} ({model_name})")
                    return model
                except Exception as e:
                    logger.warning(f"Failed to load SB3 checkpoint {path}: {e}")

            if TORCH_AVAILABLE:
                try:
                    state = torch.load(path, map_location="cpu")
                    model = REGISTRY.build_model(model_name, input_dim=input_dim)
                    if hasattr(model, "net"):
                        sd = state.get("model_state_dict", state) if isinstance(state, dict) else state
                        model.net.load_state_dict(sd, strict=False)
                        model.net.eval()
                        logger.info(f"Loaded frozen PyTorch weights from {path}")
                        return model
                except Exception as e:
                    logger.warning(f"Failed to load PyTorch checkpoint {path}: {e}")

    # 2. Check if baseline or rule-based model
    if card and card.algo_class in ("rule", "static"):
        logger.info(f"Building baseline model '{model_name}' directly from registry.")
        return REGISTRY.build_model(model_name)

    # 3. Fallback when weights_path does not exist
    logger.warning(
        f"No promoted weights found at '{weights_path}'. Running fallback MomentumBaseline "
        f"for pipeline verification. Run scripts/train_and_promote.py to train and deploy real weights."
    )
    return MomentumBaseline()


class LivePaperInferenceServer:
    def __init__(
        self,
        symbol: str = "BTC-USDT",
        weights_path: Optional[str] = None,
        model_name: Optional[str] = None,
        initial_cash: float = 10.0,
        seq_len: int = 64,
        use_synthetic_feed: bool = False,
        state_dir: str = "state",
        account_id: int = 900,
    ):
        self.symbol = symbol
        self.seq_len = seq_len
        self.tick = 0
        self.obs_buffer: Deque[np.ndarray] = deque(maxlen=seq_len)

        self.model_name = model_name or DEPLOY_CONFIG.model_name
        self.is_running = False
        self.model = load_frozen_model(
            weights_path or DEPLOY_CONFIG.frozen_model_path,
            self.model_name,
        )
        self.feature_engine = StreamingFeatureEngine()
        self.exchange = PaperExchange(
            symbol=symbol, initial_cash=initial_cash, state_dir=state_dir, account_id=account_id
        )
        self.risk = RiskGuardian(RiskLimits(
            max_position_fraction=DEPLOY_CONFIG.max_position_fraction,
            min_hold_ticks=DEPLOY_CONFIG.min_hold_ticks,
            max_daily_loss_pct=DEPLOY_CONFIG.max_daily_loss_pct,
            max_drawdown_halt=DEPLOY_CONFIG.max_drawdown_halt,
            kill_switch_path=os.path.join(state_dir, "KILL_SWITCH"),
        ))

        self.use_synthetic_feed = use_synthetic_feed
        if use_synthetic_feed:
            logger.warning("Running against SyntheticReplayFeed — NOT real market data. Logic-testing mode only.")
            self.feed = SyntheticReplayFeed(symbol=symbol)
        else:
            self.feed = BinanceLiveDepthFeed(symbol=symbol, on_update=self._on_update)

        self._init_decision_ledger(state_dir, account_id)

    @classmethod
    def from_config(cls, cfg: Any):
        """Construct LivePaperInferenceServer from DeploymentConfig."""
        server = cls(
            symbol=getattr(cfg, "symbol", "BTC-USDT"),
            weights_path=getattr(cfg, "frozen_model_path", None),
            model_name=getattr(cfg, "model_name", "PPO-DilatedCNN"),
            initial_cash=getattr(cfg, "starting_capital", 100.0),
            use_synthetic_feed=getattr(cfg, "exchange_mode", "paper") == "paper",
        )
        return server

    def get_status(self) -> Dict[str, Any]:
        """Return server status."""
        return {
            "model": self.model_name,
            "running": self.is_running,
            "symbol": self.symbol,
            "tick": self.tick,
            "equity": self.exchange.accounting.equity if hasattr(self.exchange, "accounting") else self.exchange.initial_cash,
        }

    def _init_decision_ledger(self, state_dir: str, account_id: int):
        db_dir = os.path.join(os.path.abspath(state_dir), f"child_{account_id}")
        os.makedirs(db_dir, exist_ok=True)
        self.decision_db_path = os.path.join(db_dir, "decisions.sqlite")
        conn = sqlite3.connect(self.decision_db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS decisions (
                id INTEGER PRIMARY KEY AUTOINCREMENT, tick INTEGER, timestamp REAL,
                target_frac REAL, approved INTEGER, reject_reason TEXT, equity REAL,
                mid_price REAL
            )
        """)
        try:
            conn.execute("ALTER TABLE decisions ADD COLUMN mid_price REAL")
        except Exception:
            pass
        conn.commit()
        conn.close()

    def _log_decision(self, target_frac: float, approved: bool, reason: Optional[str], mid_price: float = 0.0):
        try:
            conn = sqlite3.connect(self.decision_db_path, timeout=5)
            conn.execute(
                "INSERT INTO decisions (tick, timestamp, target_frac, approved, reject_reason, equity, mid_price) VALUES (?,?,?,?,?,?,?)",
                (self.tick, time.time(), target_frac, int(approved), reason, self.exchange.accounting.equity, mid_price),
            )
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Decision ledger write failed: {e}")

    def _think(self) -> float:
        seq = np.stack(list(self.obs_buffer), axis=0)  # (seq_len, 5)
        mean = seq.mean(axis=0)
        std = seq.std(axis=0) + 1e-8
        norm = ((seq - mean) / std).astype(np.float32)

        # Standard Gymnasium observation dict expected by SB3 & Baselines
        port_state = np.array([
            self.exchange.accounting.cash / max(self.exchange.initial_cash, 1.0),
            self.exchange.position_qty,
            self.exchange.accounting.equity / max(self.exchange.initial_cash, 1.0),
            self.exchange.accounting.max_drawdown,
        ], dtype=np.float32)

        obs = {
            "lob_sequence": norm,
            "portfolio_state": port_state,
        }

        # 1. SB3 or Baseline predict(obs) interface
        if hasattr(self.model, "predict"):
            try:
                action, _ = self.model.predict(obs, deterministic=True)
                if isinstance(action, (list, np.ndarray)):
                    return float(np.clip(action[0], -1.0, 1.0))
                return float(np.clip(action, -1.0, 1.0))
            except Exception as e:
                logger.debug(f"predict(obs) failed ({e}), checking legacy paths")

        # 2. PyTorch neural network forward pass
        if TORCH_AVAILABLE and hasattr(self.model, "net"):
            t_in = torch.from_numpy(norm).unsqueeze(0)
            with torch.no_grad():
                out = self.model.forward(t_in)
            out_arr = out.numpy() if hasattr(out, "numpy") else np.array(out)
            if out_arr.shape[-1] == 3:
                idx = int(np.argmax(out_arr, axis=-1).reshape(-1)[0])
                action_val = {0: 0.0, 1: 1.0, 2: -1.0}.get(idx, 0.0)
            else:
                action_val = float(np.mean(out_arr))
            return float(np.clip(action_val, -1.0, 1.0))

        # 3. Callable model.forward(norm)
        if hasattr(self.model, "forward"):
            try:
                out = self.model.forward(norm)
                out_arr = np.array(out)
                if out_arr.shape[-1] == 3:
                    idx = int(np.argmax(out_arr, axis=-1).reshape(-1)[0])
                    action_val = {0: 0.0, 1: 1.0, 2: -1.0}.get(idx, 0.0)
                else:
                    action_val = float(np.mean(out_arr))
                return float(np.clip(action_val, -1.0, 1.0))
            except Exception:
                pass

        return 0.0

    async def _maybe_act(self):
        if len(self.obs_buffer) < self.seq_len:
            return
        self.tick += 1
        target_frac = self._think()

        approved, reason = self.risk.check(
            target_frac=target_frac,
            current_tick=self.tick,
            equity=self.exchange.accounting.equity,
            peak_equity=self.exchange.accounting.peak_equity,
            max_drawdown=self.exchange.accounting.max_drawdown,
            book_is_stale=self.exchange.book_is_stale(self.risk.limits.max_book_staleness_sec),
        )
        mid = getattr(self.exchange, "last_mid_price", 0.0)
        self._log_decision(target_frac, approved, reason, mid_price=mid)

        if not approved:
            if reason not in ("min_hold_ticks_not_elapsed",):  # this one fires constantly by design, don't spam
                logger.info(f"Tick {self.tick}: decision REJECTED ({reason}), target_frac={target_frac:.3f}")
            return

        self.risk.record_order_submitted(self.tick)
        result = await self.exchange.submit_target_position(target_frac)
        if result.filled_qty != 0.0 or result.rejected_reason:
            logger.info(
                f"Tick {self.tick}: target={target_frac:.3f} filled={result.filled_qty:.6f} "
                f"@ {result.avg_price:.2f} fee={result.fee_paid:.4f} latency={result.latency_sec*1000:.0f}ms "
                f"reason={result.rejected_reason} equity=${self.exchange.accounting.equity:.4f}"
            )

    async def _on_update(self, kind: str, payload: Dict[str, Any]):
        if kind == "depth":
            self.exchange.on_depth_update(payload)
            feat = self.feature_engine.on_depth(payload)
            if feat is not None:
                self.obs_buffer.append(feat)
                await self._maybe_act()
        elif kind == "trade":
            self.feature_engine.on_trade(payload)

    async def run(self, duration_sec: Optional[float] = None):
        logger.info(
            f"Starting live-paper session: symbol={self.symbol} "
            f"cash=${self.exchange.initial_cash:.2f} model={DEPLOY_CONFIG.model_name} "
            f"synthetic_feed={self.use_synthetic_feed}"
        )
        try:
            if self.use_synthetic_feed:
                await self.feed.run(self._on_update, duration_sec=duration_sec)
            else:
                await self.feed.run(duration_sec=duration_sec)
        finally:
            self.exchange.close()
            logger.info(
                f"Session ended. Final equity: ${self.exchange.accounting.equity:.4f} "
                f"(started at ${self.exchange.initial_cash:.2f}). "
                f"Trades: {self.exchange.trade_count}. Total fees: ${self.exchange.total_fees_paid:.4f}. "
                f"Ledger: {self.exchange.fill_db_path} / {self.decision_db_path}"
            )


def _parse_args():
    p = argparse.ArgumentParser(description="TradeJack live-paper inference session")
    p.add_argument("--symbol", default="BTC-USDT")
    p.add_argument("--cash", type=float, default=10.0)
    p.add_argument("--duration-sec", type=float, default=None)
    p.add_argument("--weights-path", default=None)
    p.add_argument("--model-name", default=None)
    p.add_argument("--synthetic", action="store_true", help="use SyntheticReplayFeed instead of real Binance data (logic testing only)")
    p.add_argument("--state-dir", default="state")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    server = LivePaperInferenceServer(
        symbol=args.symbol,
        weights_path=args.weights_path,
        model_name=args.model_name,
        initial_cash=args.cash,
        use_synthetic_feed=args.synthetic,
        state_dir=args.state_dir,
    )
    asyncio.run(server.run(duration_sec=args.duration_sec))


# Alias for backward compatibility with scripts/run_all_checks.py
LiveInferenceServer = LivePaperInferenceServer
