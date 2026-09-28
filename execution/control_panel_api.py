"""
Control Panel API — the backend the Node/React frontend talks to.

Wraps the REAL, already-built-and-tested trading objects directly (LivePaperInferenceServer,
ContinuousTrainer, RiskGuardian) — this file adds no new trading logic of its own, only a
REST + WebSocket surface over methods/state that already exist and were verified elsewhere
in this project:
  - LivePaperInferenceServer.get_status() / .run_forever() / .stop() / .risk / .exchange
  - ContinuousTrainer.run_continuous() / .stop() / .list_pending_promotions() /
    .approve_pending_promotion()
  - RiskGuardian.state.is_halted / .state.halt_reason / .reset_halt() /
    .risk_budget_used_fraction()

SAFETY: this process can start/stop real trading and approve a model promotion that
starts placing real orders. Every mutating endpoint requires a bearer token
(TRADEJACK_API_TOKEN env var, no default — the app refuses to start without one being
set, on purpose: a control panel for live trading should never silently run with no
auth because an operator forgot to set a variable). CORS is restricted to one
configurable origin (TRADEJACK_FRONTEND_ORIGIN), not "*".

VERIFICATION STATUS: fastapi/uvicorn are not installed in the sandbox this was written
in (no network access to pip install them), so this file could not be run or hit with a
real request here. It was written by directly reading and matching the exact method
names, attribute names, and return shapes of the real objects it wraps (grep'd and
viewed in this same session, not assumed from memory) and checked with `python -m
py_compile`. Run `pip install fastapi uvicorn[standard]` and start this
(`uvicorn execution.control_panel_api:app`) in a real environment before trusting it —
in particular, verify the WebSocket broadcast loop's behavior under a real client
disconnect, which is the one thing this kind of code most often gets subtly wrong and
is hardest to reason about without actually running it.
"""

import os
import json
import time
import sqlite3
import asyncio
import logging
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Depends, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel

from execution.live_inference_server import LivePaperInferenceServer
from training.continuous_trainer import ContinuousTrainer
from scripts.deploy_config import DEPLOY_CONFIG

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] (ControlPanelAPI) %(message)s")
logger = logging.getLogger("ControlPanelAPI")

API_TOKEN = os.environ.get("TRADEJACK_API_TOKEN")
if not API_TOKEN:
    raise RuntimeError(
        "TRADEJACK_API_TOKEN is not set. This API can start/stop real trading and "
        "approve model promotions -- it refuses to start without an explicit auth "
        "token rather than silently running open. Set TRADEJACK_API_TOKEN and retry."
    )

FRONTEND_ORIGIN = os.environ.get("TRADEJACK_FRONTEND_ORIGIN", "http://localhost:5173")

app = FastAPI(title="TradeJack Control Panel API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_ORIGIN],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type"],
)

_bearer = HTTPBearer(auto_error=False)


def require_auth(creds: Optional[HTTPAuthorizationCredentials] = Depends(_bearer)) -> None:
    if creds is None or creds.credentials != API_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid or missing bearer token.")


class AppState:
    """Holds the one live server/trainer instance this process manages, and the
    background asyncio tasks running them. A single-process, single-account control
    panel -- not designed to manage multiple concurrent trading sessions."""

    def __init__(self):
        self.server: Optional[LivePaperInferenceServer] = None
        self.trainer: Optional[ContinuousTrainer] = None
        self._server_task: Optional[asyncio.Task] = None
        self._trainer_task: Optional[asyncio.Task] = None
        self.ws_clients: List[WebSocket] = []

    def is_trading_running(self) -> bool:
        return self._server_task is not None and not self._server_task.done()

    def is_training_running(self) -> bool:
        return self._trainer_task is not None and not self._trainer_task.done()


state = AppState()


def _build_server() -> LivePaperInferenceServer:
    """Constructs the server from DEPLOY_CONFIG, exactly the way scripts/genesis_prime.py
    does for its paper/testnet/live modes -- this file does not invent a different
    construction path."""
    return LivePaperInferenceServer.from_config(DEPLOY_CONFIG)


# ── Status ──────────────────────────────────────────────────────────────────────

def _collect_status() -> Dict[str, Any]:
    """
    Merges LivePaperInferenceServer.get_status() with the risk/exchange/training state
    the control panel actually needs to display, reading directly from the real
    objects' already-confirmed attributes rather than adding new server-side state of
    its own. Defensive throughout: a control panel must never itself crash because one
    field was momentarily unavailable mid-construction/mid-shutdown.
    """
    if state.server is None:
        return {
            "trading_running": False,
            "training_running": False,
            "symbol": getattr(DEPLOY_CONFIG, "symbol", None),
            "exchange_mode": getattr(DEPLOY_CONFIG, "exchange_mode", None),
        }

    try:
        base = state.server.get_status()
    except Exception as e:
        logger.error(f"get_status() failed: {e}")
        base = {}

    risk = state.server.risk
    exchange = state.server.exchange

    out = {
        **base,
        "trading_running": state.is_trading_running(),
        "training_running": state.is_training_running(),
        "exchange_mode": state.server.exchange_mode,
        "risk": {
            "is_halted": risk.state.is_halted,
            "halt_reason": risk.state.halt_reason,
            "current_equity": risk.state.current_equity,
            "peak_equity": risk.state.peak_equity,
            "risk_budget_used_fraction": _safe_call(risk.risk_budget_used_fraction, default=None),
            "max_daily_loss_pct": risk.limits.max_daily_loss_pct,
            "max_drawdown_halt": risk.limits.max_drawdown_halt,
        },
        "position": {
            "qty": getattr(exchange, "position_qty", None),
            "cash": getattr(getattr(exchange, "accounting", None), "cash", None),
            "max_drawdown": getattr(getattr(exchange, "accounting", None), "max_drawdown", None),
        },
        "trade_count": getattr(exchange, "trade_count", None),
        "total_fees_paid": getattr(exchange, "total_fees_paid", None),
    }

    if state.trainer is not None:
        out["training"] = {
            "cycle_count": state.trainer.cycle_count,
            "promotion_count": state.trainer.promotion_count,
        }

    out["portfolio"] = _collect_portfolio()

    return out


def _collect_portfolio() -> Dict[str, Any]:
    """Reads execution/portfolio_orchestrator.py's persisted allocation state --
    same file-based, no-live-instance-needed pattern as RiskGuardian's
    ACTIVE_HALT.json (see PortfolioOrchestrator.read_latest_allocation()'s own
    docstring). Deliberately NOT constructed from a live PortfolioOrchestrator
    instance held by this process: nothing in this codebase runs one yet (see
    that module's own "what this deliberately does NOT do" section) -- an
    orchestrator, if and when one runs (as its own scheduled loop, wherever
    that ends up living), writes this file independently, and the control
    panel just displays whatever's there. Returns an explicit empty shape,
    never 404/error, when nothing has been persisted -- that's the normal
    state for a deployment that hasn't wired an orchestrator loop up yet, not
    a fault."""
    from execution.portfolio_orchestrator import PortfolioOrchestrator
    state_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "state", "PORTFOLIO_ALLOCATION.json")
    data = _safe_call(lambda: PortfolioOrchestrator.read_latest_allocation(state_path), default=None)
    if data is None:
        return {"generated_at": None, "decisions": [], "correlation_matrix": {}}
    return data


def _safe_call(fn, default=None):
    try:
        return fn()
    except Exception as e:
        logger.debug(f"Safe call failed ({fn}): {e}")
        return default


@app.get("/api/status")
def get_status(_: None = Depends(require_auth)) -> Dict[str, Any]:
    return _collect_status()


# ── Trading control ─────────────────────────────────────────────────────────────

@app.post("/api/trading/start")
async def start_trading(_: None = Depends(require_auth)) -> Dict[str, Any]:
    if state.is_trading_running():
        raise HTTPException(status_code=409, detail="Trading is already running.")

    if state.server is None:
        state.server = _build_server()

    state._server_task = asyncio.create_task(state.server.run_forever())
    logger.info("Trading started via control panel.")
    return {"ok": True, "status": _collect_status()}


@app.post("/api/trading/stop")
async def stop_trading(_: None = Depends(require_auth)) -> Dict[str, Any]:
    if not state.is_trading_running():
        raise HTTPException(status_code=409, detail="Trading is not running.")

    state.server.stop()
    logger.info("Trading stop requested via control panel.")
    return {"ok": True, "status": _collect_status()}


@app.post("/api/training/start")
async def start_training(_: None = Depends(require_auth)) -> Dict[str, Any]:
    if state.is_training_running():
        raise HTTPException(status_code=409, detail="Continuous training is already running.")
    if state.server is None:
        raise HTTPException(status_code=409, detail="Start trading first -- the trainer needs a live inference server to promote into.")

    if state.trainer is None:
        state.trainer = ContinuousTrainer(
            data_store_dir=os.path.join(os.path.dirname(os.path.dirname(__file__)), "data_store"),
            # state_dir anchored the same way, for the same reason
            # data_store_dir already was -- see genesis_prime.py's matching
            # comment and ContinuousTrainer.__init__'s deployed_model_path
            # fix; this was the other real call site with the same gap.
            state_dir=os.path.join(os.path.dirname(os.path.dirname(__file__)), "state"),
            inference_server=state.server,
            # MUST match the live server's own symbol -- ContinuousTrainer defaults
            # to "BTC-USDT" if not given, which silently trained the wrong
            # instrument whenever DEPLOY_CONFIG.symbol was anything else. This was
            # harmless while every deployment traded BTC-USDT by convention; it
            # stopped being harmless the moment this project gained real FX/OANDA
            # support (execution/oanda_adapter.py) and DEPLOY_CONFIG.symbol could
            # legitimately be "EURUSD" while training silently kept optimizing a
            # BTC-USDT model. Caught by reviewing this call against
            # ContinuousTrainer's actual default, not by a live incident.
            symbol=state.server.symbol,
        )

    state._trainer_task = asyncio.create_task(state.trainer.run_continuous())
    logger.info("Continuous training started via control panel.")
    return {"ok": True, "status": _collect_status()}


@app.post("/api/training/stop")
async def stop_training(_: None = Depends(require_auth)) -> Dict[str, Any]:
    if not state.is_training_running():
        raise HTTPException(status_code=409, detail="Continuous training is not running.")

    state.trainer.stop()
    logger.info("Continuous training stop requested via control panel.")
    return {"ok": True, "status": _collect_status()}


# ── Risk ────────────────────────────────────────────────────────────────────────

@app.post("/api/risk/halt/reset")
def reset_halt(_: None = Depends(require_auth)) -> Dict[str, Any]:
    if state.server is None:
        raise HTTPException(status_code=409, detail="No server constructed yet.")
    if not state.server.risk.state.is_halted:
        raise HTTPException(status_code=409, detail="Risk guardian is not currently halted.")

    state.server.risk.reset_halt()
    logger.warning("Risk halt manually reset via control panel.")
    return {"ok": True, "status": _collect_status()}


# ── Pending promotions ───────────────────────────────────────────────────────────

@app.get("/api/promotions/pending")
def list_pending_promotions(_: None = Depends(require_auth)) -> Dict[str, Any]:
    if state.trainer is None:
        return {"pending": {}}
    return {"pending": state.trainer.list_pending_promotions()}


class ApprovePromotionRequest(BaseModel):
    agent_id: int


@app.post("/api/promotions/{agent_id}/approve")
def approve_promotion(agent_id: int, _: None = Depends(require_auth)) -> Dict[str, Any]:
    if state.trainer is None:
        raise HTTPException(status_code=409, detail="No trainer running -- nothing to approve.")

    ok = state.trainer.approve_pending_promotion(agent_id)
    if not ok:
        raise HTTPException(status_code=404, detail=f"No pending promotion found for agent_id={agent_id}.")
    logger.warning(f"Promotion for agent_id={agent_id} approved via control panel.")
    return {"ok": True}


# ── Trades ──────────────────────────────────────────────────────────────────────

@app.get("/api/trades/recent")
def recent_trades(limit: int = 50, _: None = Depends(require_auth)) -> Dict[str, Any]:
    """
    Reads directly from the exchange's own fills.sqlite (schema confirmed by reading
    execution/paper_exchange.py's CREATE TABLE statement, not guessed) -- this endpoint
    adds no new persistence, it's a read-only window onto the fill ledger the exchange
    already writes for every order attempt (filled or rejected).
    """
    if state.server is None or not hasattr(state.server.exchange, "fill_db_path"):
        return {"trades": []}

    db_path = state.server.exchange.fill_db_path
    if not os.path.exists(db_path):
        return {"trades": []}

    try:
        conn = sqlite3.connect(db_path, timeout=5)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """SELECT timestamp, symbol, side, requested_qty, filled_qty, avg_price,
                      fee_paid, fully_filled, rejected_reason, resulting_equity
               FROM fills ORDER BY id DESC LIMIT ?""",
            (limit,),
        ).fetchall()
        conn.close()
        return {"trades": [dict(r) for r in rows]}
    except Exception as e:
        logger.error(f"Failed to read fills.sqlite: {e}")
        raise HTTPException(status_code=500, detail="Failed to read trade history.")


@app.get("/api/equity/history")
def equity_history(limit: int = 200, _: None = Depends(require_auth)) -> Dict[str, Any]:
    """Equity trail sourced from fills.resulting_equity -- the actual equity value
    recorded at each real order attempt, not a separately-sampled series that could
    drift from what the exchange itself booked."""
    if state.server is None or not hasattr(state.server.exchange, "fill_db_path"):
        return {"points": []}

    db_path = state.server.exchange.fill_db_path
    if not os.path.exists(db_path):
        return {"points": []}

    try:
        conn = sqlite3.connect(db_path, timeout=5)
        rows = conn.execute(
            "SELECT timestamp, resulting_equity FROM fills ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        conn.close()
        points = [{"timestamp": t, "equity": e} for t, e in reversed(rows)]
        return {"points": points}
    except Exception as e:
        logger.error(f"Failed to read equity history: {e}")
        raise HTTPException(status_code=500, detail="Failed to read equity history.")


# ── Live WebSocket ──────────────────────────────────────────────────────────────

@app.websocket("/ws/live")
async def ws_live(websocket: WebSocket):
    """
    Broadcasts a status snapshot once per second to every connected client. Auth here
    is via a `token` query param (`/ws/live?token=...`) since browser WebSocket clients
    cannot set a custom Authorization header on the handshake request -- this is the
    standard workaround, not an oversight; the token is still required and still
    checked against the same API_TOKEN.

    Each client gets its own send loop so one slow/dead connection can't block
    broadcasts to the others; a failed send removes that client rather than crashing
    the loop for everyone.
    """
    token = websocket.query_params.get("token")
    if token != API_TOKEN:
        await websocket.close(code=4401)
        return

    await websocket.accept()
    state.ws_clients.append(websocket)
    logger.info(f"WS client connected ({len(state.ws_clients)} total).")

    try:
        while True:
            try:
                payload = _collect_status()
                await websocket.send_text(json.dumps(payload))
            except Exception as e:
                logger.debug(f"WS send failed, dropping client: {e}")
                break
            await asyncio.sleep(1.0)
    except WebSocketDisconnect:
        pass
    finally:
        if websocket in state.ws_clients:
            state.ws_clients.remove(websocket)
        logger.info(f"WS client disconnected ({len(state.ws_clients)} remaining).")


@app.get("/api/health")
def health() -> Dict[str, str]:
    """Unauthenticated on purpose -- a load balancer / process supervisor health
    check shouldn't need the trading API token."""
    return {"status": "ok"}
