"""
SleeveConfig: one config per trading "sleeve" (a strategy running at its own
decision cadence, holding period, risk budget, and capital slice).

Read this alongside multi_sleeve_orchestrator.py's docstring for the honest
framing of what "HFT" means here — retail public WebSocket + Python asyncio,
not colocated microsecond-scale HFT.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SleeveConfig:
    name: str                          # "hft_scalper", "position_swing", etc — used in logs and account_id derivation
    model_name: str                    # one of REGISTRY's model cards, e.g. "DuelingDQN", "PPO-Transformer"
    capital_fraction: float            # this sleeve's slice of total session capital, in [0, 1]; sleeves should sum to <= 1.0
    decision_interval_ticks: int       # 1 = act on every depth update (~100ms cadence); larger = slower/calmer
    min_hold_ticks: int                # in units of THIS sleeve's own decisions, not raw depth updates
    max_position_fraction: float       # of THIS sleeve's own capital slice, not total session capital
    seq_len: int = 64                  # observation window length fed to the model
    max_orders_per_minute: int = 6
    account_id_offset: int = 0         # combined with a base account_id by the orchestrator to keep ledgers separate

    def describe(self) -> str:
        return (
            f"{self.name}: model={self.model_name} capital={self.capital_fraction*100:.0f}% "
            f"decision_every={self.decision_interval_ticks} ticks min_hold={self.min_hold_ticks} "
            f"max_pos={self.max_position_fraction*100:.0f}%"
        )


# Two reference sleeve configs matching the model-family recommendation from
# the earlier architecture review: DQN (cheap, fast inference) for the
# fast-reacting sleeve, PPO-Transformer (higher capacity, longer context) for
# the slower position sleeve. Tune capital_fraction and the tick counts to
# your actual measured update rate before using these for anything real —
# the tick counts below assume Binance's depth@100ms stream (~10 updates/sec).
DEFAULT_HFT_SLEEVE = SleeveConfig(
    name="hft_scalper",
    model_name="DuelingDQN",
    capital_fraction=0.4,
    decision_interval_ticks=1,        # react on every ~100ms update
    min_hold_ticks=20,                # ~2s minimum hold between flips, in decision units (== update units here)
    max_position_fraction=0.3,        # smaller size per trade — many trades, tight risk per trade
    seq_len=32,                       # shorter context: recent order flow matters more than distant history
    max_orders_per_minute=20,
    account_id_offset=0,
)

DEFAULT_POSITION_SLEEVE = SleeveConfig(
    name="position_swing",
    model_name="PPO-Transformer",
    capital_fraction=0.6,
    decision_interval_ticks=300,      # ~ every 30s at 100ms/update — tune to your actual measured rate, don't assume
    min_hold_ticks=20,                # 20 DECISIONS at 30s cadence = ~10 minutes minimum hold, not 20 raw updates
    max_position_fraction=0.7,        # larger size per trade — few trades, held longer, sized with more conviction
    seq_len=128,                      # longer context: trend/regime signal needs more history than a scalp does
    max_orders_per_minute=3,
    account_id_offset=1,
)
