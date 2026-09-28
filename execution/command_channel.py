"""
Command Channel — file-based control commands for the trading loop.

DESIGN CORRECTION (full-design review): execution/control_panel_api.py originally
held live in-process references to LivePaperInferenceServer/ContinuousTrainer and
started them via asyncio.create_task() inside the API process itself -- meaning the
API process and the trading process were the same process. This was inconsistent
with a pattern this exact project had ALREADY established and validated twice over:
- execution/risk_guardian.py's kill_switch_path -- "touch this file to halt trading
  immediately" -- polled every tick via a simple os.path.exists() check.
- training/continuous_trainer.py's pending_promotions/*.json files -- a durable,
  file-based record of a queued action, consumed by whichever process gets around
  to it.

Both already prove the right shape for a control surface over live trading: the
thing issuing a command and the thing executing it should be different processes,
so a crash or restart of one can never affect the other. control_panel_api.py now
follows that same shape instead of introducing a third, tighter-coupled pattern of
its own -- it only ever writes/removes command files and reads status from the
same SQLite ledgers dashboard/telemetry_server.py already reads directly; it holds
no live reference to a running server or trainer at all.

Commands are plain files under {state_dir}/commands/, one file per pending command,
named after the command type. The trading loop (see
execution/live_inference_server.py's run_forever() / scripts/genesis_prime.py's
run_live()) polls this directory once per cycle via poll_and_execute() and deletes
each file after acting on it -- so a command file's mere existence is itself the
durable queue, the same way KILL_SWITCH's mere existence is the halt signal.
"""

import os
import json
import logging
from typing import Optional, Callable, Dict, Any

logger = logging.getLogger("CommandChannel")

START_TRADING = "start_trading"
STOP_TRADING = "stop_trading"
START_TRAINING = "start_training"
STOP_TRAINING = "stop_training"
RESET_HALT = "reset_halt"
APPROVE_PROMOTION = "approve_promotion"  # payload: {"agent_id": <int>}


def _commands_dir(state_dir: str) -> str:
    return os.path.join(state_dir, "commands")


def issue_command(state_dir: str, command: str, payload: Optional[Dict[str, Any]] = None) -> str:
    """
    Writes a command file. Returns the path written. Uses write-to-temp-then-rename
    (the same atomic pattern used throughout this project's other file-based state --
    lob_collector.py's partition flushes, scenario_injection.py's scenario writes)
    so the trading loop's poller never observes a half-written command file.
    """
    commands_dir = _commands_dir(state_dir)
    os.makedirs(commands_dir, exist_ok=True)
    path = os.path.join(commands_dir, f"{command}.cmd")
    tmp_path = path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(payload or {}, f)
    os.replace(tmp_path, path)
    logger.info(f"Command issued: {command} ({payload or {}}) -> {path}")
    return path


def poll_and_execute(state_dir: str, handlers: Dict[str, Callable[[Dict[str, Any]], None]]) -> None:
    """
    Called once per cycle by the trading loop (or continuous trainer loop) itself --
    NOT by the API process. Reads every pending command file, calls the matching
    handler with its payload, and removes the file whether the handler succeeded or
    raised -- a broken command must never be able to wedge the queue by being
    retried forever, and must never crash the caller's own loop (any handler
    exception is logged and swallowed here, exactly like this project's other
    resilience boundaries -- e.g. training/continuous_trainer.py's
    _bridge_live_data()).

    handlers: {command_name: callable(payload_dict)}. A command file with no
    matching handler is logged and removed, not silently ignored forever.
    """
    commands_dir = _commands_dir(state_dir)
    if not os.path.isdir(commands_dir):
        return

    for fname in os.listdir(commands_dir):
        if not fname.endswith(".cmd"):
            continue
        command = fname[: -len(".cmd")]
        path = os.path.join(commands_dir, fname)

        try:
            with open(path, "r") as f:
                payload = json.load(f)
        except Exception as e:
            logger.error(f"Failed to read command file {path}: {e}")
            payload = {}

        handler = handlers.get(command)
        if handler is None:
            logger.warning(f"No handler registered for command '{command}' -- discarding.")
        else:
            try:
                handler(payload)
                logger.info(f"Executed command: {command} ({payload})")
            except Exception as e:
                logger.error(f"Command '{command}' handler raised ({e}) -- command still consumed, not retried.")

        try:
            os.remove(path)
        except FileNotFoundError:
            pass
