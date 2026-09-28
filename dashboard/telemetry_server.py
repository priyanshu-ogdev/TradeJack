"""
TradeJack Telemetry Dashboard: a fully separate process from the trading
loop, reading directly from the SQLite ledgers under `state_dir` that
execution/paper_exchange.py, execution/live_inference_server.py, and
swarm/training_progress_ledger.py already write to.

WHY A SEPARATE PROCESS, READING FILES, RATHER THAN AN IN-PROCESS REFERENCE:
The trading loop is asyncio-based (paper_exchange.py, live_inference_server.py,
multi_sleeve_orchestrator.py). Flask's built-in dev server is synchronous.
Trying to run both in one process means either blocking the trading loop's
event loop with Flask's request handling, or building a bridge between them
that adds real complexity for no real benefit. Every piece of state the
dashboard needs (fills, decisions, equity, training progress) already lands
in SQLite specifically so it can be read from anywhere -- so the dashboard
just reads the same files the trading loop writes, completely decoupled.
This is also the right shape for a real deployment: the trading loop and the
dashboard can run as separate services, restart independently, and one
crashing does not take the other down.

SECURITY, taken as seriously as everywhily else real-money-adjacent in this
project:
  - Binds to 127.0.0.1 by default. Change DASHBOARD_HOST only if you have a
    reverse proxy with real authentication and TLS in front of this --
    there is no built-in HTTPS here.
  - Every endpoint that DOES something (kill switch, revealing a deposit
    address) requires a bearer token (DASHBOARD_TOKEN env var, or a random
    one generated at startup and printed once). Plain GET status/chart data
    endpoints do not require it, on the assumption that "local machine only"
    is the actual security boundary for those -- if you ever expose this
    dashboard beyond localhost, put auth in front of everything, not just
    the action endpoints.
  - The exchange adapter's real API keys are never sent to the browser --
    every Binance-touching call happens server-side in this file; the
    frontend only ever sees the JSON results.
  - There is deliberately no "withdraw funds" endpoint anywhere in this file,
    matching execution/exchange_adapter.py's own decision not to implement
    withdrawal at all. Adding funds is something you do directly on Binance;
    this dashboard can only show you where to send them (a deposit address)
    and what's already there (balance).
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import time
import json
import sqlite3
import asyncio
import secrets
import logging
from functools import wraps
from typing import Any, Dict, List, Optional

from flask import Flask, jsonify, request, render_template

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] (Dashboard) %(message)s")
logger = logging.getLogger("Dashboard")

app = Flask(__name__)

STATE_DIR = os.path.abspath(os.environ.get("TRADEJACK_STATE_DIR", "state"))
DASHBOARD_TOKEN = os.environ.get("DASHBOARD_TOKEN")
if not DASHBOARD_TOKEN:
    DASHBOARD_TOKEN = secrets.token_urlsafe(24)
    logger.warning(
        f"DASHBOARD_TOKEN not set -- generated one for this session: {DASHBOARD_TOKEN}\n"
        f"Set DASHBOARD_TOKEN in your environment to keep it stable across restarts. "
        f"Any request to an action endpoint (kill switch, deposit address) needs "
        f"'Authorization: Bearer {DASHBOARD_TOKEN}'."
    )

_exchange_adapter = None  # lazily connected real BinanceSpotAdapter, if configured


def require_token(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        auth = request.headers.get("Authorization", "")
        token = auth[len("Bearer "):] if auth.startswith("Bearer ") else ""
        if not secrets.compare_digest(token, DASHBOARD_TOKEN):
            return jsonify({"error": "unauthorized"}), 401
        return f(*args, **kwargs)
    return wrapper


def _db(path: str) -> Optional[sqlite3.Connection]:
    if not os.path.exists(path):
        return None
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def _child_dir(account_id: int) -> str:
    return os.path.join(STATE_DIR, f"child_{account_id}")


# --------------------------------------------------------------------- pages
@app.route("/")
def index():
    return render_template("index.html", state_dir=STATE_DIR)


# ---------------------------------------------------------------- read APIs
@app.route("/api/status")
def api_status():
    """Aggregates whatever accounts/agents actually have state on disk --
    works whether this is a single-server deployment or a multi-sleeve one,
    since both write the same child_{account_id}/ledger.sqlite shape."""
    accounts = []
    if os.path.isdir(STATE_DIR):
        for name in sorted(os.listdir(STATE_DIR)):
            if not name.startswith("child_"):
                continue
            account_id = name[len("child_"):]
            conn = _db(os.path.join(STATE_DIR, name, "ledger.sqlite"))
            if conn is None:
                continue
            row = conn.execute("SELECT * FROM portfolio_state ORDER BY tick_id DESC LIMIT 1").fetchone()
            conn.close()
            if row:
                accounts.append({
                    "account_id": account_id,
                    "equity": row["equity"],
                    "cash": row["cash"],
                    "tick": row["tick_id"],
                    "market_timestamp": row["market_timestamp"],
                })

    kill_switch_engaged = os.path.exists(os.path.join(STATE_DIR, "KILL_SWITCH"))
    return jsonify({
        "state_dir": STATE_DIR,
        "kill_switch_engaged": kill_switch_engaged,
        "accounts": accounts,
        "server_time": time.time(),
    })


@app.route("/api/price-series")
def api_price_series():
    """Real price line to plot underneath the entry/exit trade markers --
    without this, the dashboard could only show isolated trade points with
    nothing connecting them."""
    account_id = request.args.get("account_id", "900")
    limit = min(int(request.args.get("limit", 2000)), 20000)
    conn = _db(os.path.join(_child_dir(int(account_id)), "fills.sqlite"))
    if conn is None:
        return jsonify({"points": []})
    rows = conn.execute(
        "SELECT timestamp, mid_price FROM price_samples ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    return jsonify({"points": [{"t": r["timestamp"], "price": r["mid_price"]} for r in reversed(rows)]})


@app.route("/api/trades")
def api_trades():
    """Entry/exit markers: every real fill for one account, from
    paper_exchange.py's / exchange_adapter's fills.sqlite."""
    account_id = request.args.get("account_id", "900")
    limit = min(int(request.args.get("limit", 300)), 2000)
    conn = _db(os.path.join(_child_dir(int(account_id)), "fills.sqlite"))
    if conn is None:
        return jsonify({"trades": []})
    rows = conn.execute(
        "SELECT timestamp, side, filled_qty, avg_price, fee_paid, fully_filled, rejected_reason "
        "FROM fills WHERE filled_qty != 0 ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    trades = [dict(r) for r in reversed(rows)]
    return jsonify({"trades": trades})


@app.route("/api/equity-curve")
def api_equity_curve():
    account_id = request.args.get("account_id", "900")
    limit = min(int(request.args.get("limit", 2000)), 20000)
    conn = _db(os.path.join(_child_dir(int(account_id)), "ledger.sqlite"))
    if conn is None:
        return jsonify({"points": []})
    rows = conn.execute(
        "SELECT market_timestamp, equity FROM portfolio_state ORDER BY tick_id DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    points = [{"t": r["market_timestamp"], "equity": r["equity"]} for r in reversed(rows)]
    return jsonify({"points": points})


@app.route("/api/agents")
def api_agents():
    """Distinct agent_ids known to the training progress ledger, so the
    frontend can populate a selector for the RL progress chart."""
    conn = _db(os.path.join(STATE_DIR, "training_progress.sqlite"))
    if conn is None:
        return jsonify({"agents": []})
    rows = conn.execute(
        "SELECT DISTINCT agent_id, label, model_name FROM cycles ORDER BY agent_id"
    ).fetchall()
    conn.close()
    return jsonify({"agents": [dict(r) for r in rows]})


@app.route("/api/training-progress")
def api_training_progress():
    """RL agent progress: Sortino/equity per cycle, plus promotion and
    plasticity-reset events to mark on the same timeline."""
    from swarm.training_progress_ledger import TrainingProgressLedger
    agent_id = request.args.get("agent_id")
    ledger = TrainingProgressLedger(state_dir=STATE_DIR)
    cycles = ledger.get_cycles(agent_id=agent_id, limit=1000)
    events = ledger.get_events(agent_id=agent_id, limit=500)
    return jsonify({"cycles": cycles, "events": events})


@app.route("/api/decisions")
def api_decisions():
    """Recent risk-guardian decisions (approved/rejected + reason) for one
    account -- useful to see WHY the model isn't trading, not just that it
    isn't."""
    account_id = request.args.get("account_id", "900")
    limit = min(int(request.args.get("limit", 200)), 2000)
    conn = _db(os.path.join(_child_dir(int(account_id)), "decisions.sqlite"))
    if conn is None:
        return jsonify({"decisions": []})
    rows = conn.execute(
        "SELECT tick, timestamp, target_frac, approved, reject_reason, equity FROM decisions ORDER BY id DESC LIMIT ?",
        (limit,),
    ).fetchall()
    conn.close()
    return jsonify({"decisions": [dict(r) for r in reversed(rows)]})


# -------------------------------------------------------- exchange (Binance)
def _get_exchange_adapter():
    """Lazily constructs a BinanceSpotAdapter from env-configured credentials.
    Returns None (not an error) if not configured -- a paper-only deployment
    simply won't have a real balance/deposit section, which is fine."""
    global _exchange_adapter
    if _exchange_adapter is not None:
        return _exchange_adapter
    try:
        from execution.exchange_adapter import BinanceSpotAdapter
        testnet = os.environ.get("EXCHANGE_MODE", "paper") != "live"
        adapter = BinanceSpotAdapter(testnet=testnet)
        if not adapter.api_key or not adapter.secret_value:
            return None
        _exchange_adapter = adapter
        return adapter
    except Exception as e:
        logger.warning(f"Exchange adapter not available: {e}")
        return None


def _run_async(coro):
    return asyncio.run(coro)


@app.route("/api/balance")
def api_balance():
    """Real Binance balance if credentials are configured; otherwise the
    paper account's cash/equity from the same ledger /api/status already
    reads. Read-only either way -- no auth required for the same reason
    /api/status doesn't need it (local-machine security boundary)."""
    adapter = _get_exchange_adapter()
    if adapter is None:
        return jsonify({"mode": "paper", "note": "No real exchange credentials configured -- showing paper accounts only via /api/status."})

    async def _fetch():
        if not adapter.is_connected():
            await adapter.connect()
        return await adapter.get_all_balances()

    try:
        balances = _run_async(_fetch())
        return jsonify({"mode": "testnet" if adapter.testnet else "live", "balances": balances})
    except Exception as e:
        logger.error(f"Balance fetch failed: {e}")
        return jsonify({"error": str(e)}), 502


@app.route("/api/deposit-address", methods=["POST"])
@require_token
def api_deposit_address():
    """The actual answer to 'add funds': shows where to send them. Cannot
    pull funds in on its own -- token-gated anyway since it's still a real
    call against your real account, consistent with every other action
    endpoint here."""
    adapter = _get_exchange_adapter()
    if adapter is None:
        return jsonify({"error": "No real exchange credentials configured."}), 400
    asset = (request.get_json(silent=True) or {}).get("asset", "USDT")

    async def _fetch():
        if not adapter.is_connected():
            await adapter.connect()
        return await adapter.get_deposit_address(asset)

    try:
        result = _run_async(_fetch())
        if result is None:
            return jsonify({"error": f"Could not fetch a deposit address for {asset}."}), 502
        return jsonify(result)
    except Exception as e:
        logger.error(f"Deposit address fetch failed: {e}")
        return jsonify({"error": str(e)}), 502


# ----------------------------------------------------------------- actions
@app.route("/api/kill-switch", methods=["POST"])
@require_token
def api_kill_switch():
    """Engages or disengages the kill switch every RiskGuardian instance
    already polls for (execution/risk_guardian.py). This dashboard doesn't
    need its own halt mechanism -- it just writes/removes the same file the
    trading loop was already built to check."""
    body = request.get_json(silent=True) or {}
    engage = body.get("engage", True)
    path = os.path.join(STATE_DIR, "KILL_SWITCH")
    if engage:
        os.makedirs(STATE_DIR, exist_ok=True)
        with open(path, "w") as f:
            f.write(f"engaged via dashboard at {time.time()}\n")
        logger.critical("KILL SWITCH ENGAGED via dashboard.")
    else:
        if os.path.exists(path):
            os.remove(path)
        logger.warning("Kill switch disengaged via dashboard.")
    return jsonify({"kill_switch_engaged": engage})


# --------------------------------------------------------------- control APIs
#
# FULL-DESIGN-REVIEW ADDITION: these follow the exact same principle
# api_kill_switch() above already established -- this dashboard process never
# executes a control action itself, it only writes a durable command file that
# whichever process is actually running the trading loop polls and acts on
# (see execution/command_channel.py, and scripts/genesis_prime.py's run_live()
# for where it's polled). This keeps the dashboard and the trading loop as
# fully independent processes: restarting either one never affects the other,
# and a command that arrives while the trading loop happens to be down simply
# waits in the commands/ directory until it comes back up, rather than being
# lost or requiring the dashboard to hold a live reference to it.
#
# A separate FastAPI backend (execution/control_panel_api.py) was previously
# built for these same actions, holding an in-process reference to a live
# LivePaperInferenceServer/ContinuousTrainer and starting them via
# asyncio.create_task() inside the API process itself. That was a real design
# mistake, found during a full-design review: it coupled the control surface
# to the trading process's lifetime in exactly the way api_kill_switch()
# above had already deliberately avoided. That file is now deprecated in
# favor of these routes -- see its own module docstring for the full
# explanation and the file-based alternative it now also offers.
from execution.command_channel import issue_command, START_TRADING, STOP_TRADING, START_TRAINING, STOP_TRAINING, RESET_HALT, APPROVE_PROMOTION


@app.route("/api/trading/start", methods=["POST"])
@require_token
def api_start_trading():
    issue_command(STATE_DIR, START_TRADING)
    return jsonify({"ok": True, "command": START_TRADING})


@app.route("/api/trading/stop", methods=["POST"])
@require_token
def api_stop_trading():
    issue_command(STATE_DIR, STOP_TRADING)
    return jsonify({"ok": True, "command": STOP_TRADING})


@app.route("/api/training/start", methods=["POST"])
@require_token
def api_start_training():
    issue_command(STATE_DIR, START_TRAINING)
    return jsonify({"ok": True, "command": START_TRAINING})


@app.route("/api/training/stop", methods=["POST"])
@require_token
def api_stop_training():
    issue_command(STATE_DIR, STOP_TRAINING)
    return jsonify({"ok": True, "command": STOP_TRAINING})


@app.route("/api/risk/halt/reset", methods=["POST"])
@require_token
def api_reset_halt():
    issue_command(STATE_DIR, RESET_HALT)
    return jsonify({"ok": True, "command": RESET_HALT})


@app.route("/api/promotions/<int:agent_id>/approve", methods=["POST"])
@require_token
def api_approve_promotion(agent_id: int):
    issue_command(STATE_DIR, APPROVE_PROMOTION, {"agent_id": agent_id})
    return jsonify({"ok": True, "command": APPROVE_PROMOTION, "agent_id": agent_id})


if __name__ == "__main__":
    host = os.environ.get("DASHBOARD_HOST", "127.0.0.1")
    port = int(os.environ.get("DASHBOARD_PORT", "5000"))
    logger.info(f"Starting TradeJack dashboard on http://{host}:{port} (state_dir={STATE_DIR})")
    app.run(host=host, port=port, debug=False)
