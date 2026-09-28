"""
Scenario Injection — Phase 3 of the upgrade plan ("more scenarios, more resilience,
where it actually reaches the deployed model").

data_forge/synthetic_diffusion.py's SyntheticDiffusionEngine.generate_black_swan_scenario()
was fully built but never called by anything in the training pipeline. Traced why:
it writes to data_store/synthetic/{symbol}/{name}.parquet, but TradeJackLOBEnv (the
env every active training path actually uses) only ever scans
data_store/processed/{symbol}/physics/ — a completely different, unscanned directory.
That's the actual root cause of the orphaning, not just "nobody called it yet."

Two ways to fix that:
  (a) Teach TradeJackLOBEnv to also scan data_store/synthetic/ — touches the core RL
      env, which is untestable in this sandbox (no gymnasium/torch) and would change
      behavior for every training path, not just the tournament.
  (b) Materialize each scenario directly into data_store/processed/{symbol}/physics/
      at a clearly reserved, unambiguous, synthetic date — zero changes needed to
      TradeJackLOBEnv at all.

This module does (b) — the minimal-blast-radius option, consistent with how the
rest of this project's fixes have preferred reusing existing, already-scanned
infrastructure over widening what a shared core component reads.

Reserved date range: 2099-01-DD (up to 31 slots). Chosen to be unambiguously
synthetic (no real market data will ever carry this date), and trivially
greppable/removable as a block (`data_store/processed/{symbol}/physics/2099/`).

Also fixed while building this (found during Phase 3 review, not assumed correct):
generate_black_swan_scenario()'s underlying process only ever varied by
volatility_multiplier — jump direction was hardcoded negative, so every "regime"
was really the same one-sided-crash pattern at different intensities, not a
genuinely different shape. SyntheticDiffusionEngine now takes jump_direction
(-1/+1/0) and jump_probability, and the DEFAULT_REGIMES below actually use that to
produce a flash crash, a melt-up squeeze, and a two-sided high-volatility regime as
real, structurally different scenarios — not the same shock relabeled three times.

VERIFICATION STATUS: no polars/pydantic_settings in this sandbox (same limitation
as every other data_forge/training module in this project). The pure-numpy
generation math (SyntheticDiffusionEngine._generate_scenario_arrays, including the
new jump_direction/jump_probability parameters) was verified directly by execution
— confirmed crash/squeeze/two-sided regimes actually produce structurally different
price paths, and that default arguments reproduce the exact prior hardcoded
behavior. The file-writing and idempotency logic in this module is verified by
source tracing and py_compile only, not executed — do a real run with polars
installed before trusting the on-disk output.
"""

import os
import logging
from typing import List, Optional, Tuple

from data_forge.config import config
from data_forge.synthetic_diffusion import SyntheticDiffusionEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] (ScenarioInjection) %(message)s")
logger = logging.getLogger("ScenarioInjection")

RESERVED_SYNTHETIC_YEAR = 2099

# (day_of_month, scenario_name, volatility_multiplier, jump_direction, jump_probability,
#  spoof_kwargs)
# Five structurally different regimes, not five intensities of the same one:
#   flash_crash          -- sharp one-sided decline, matches the prior (pre-fix) behavior
#   melt_up_squeeze       -- sharp one-sided rally (short squeeze / FOMO spike shape)
#   two_sided_high_vol    -- frequent jumps in either direction, no directional bias
#                            (a genuinely uncertain, choppy/whipsaw-prone regime)
#   liquidity_vacuum      -- rare but very large jumps (low jump_probability, high
#                            volatility_multiplier) -- approximates a thin-book,
#                            gap-prone regime via extreme VPIN/volume spikes on the
#                            rare jump ticks, since this generator's output columns
#                            have no direct bid-ask spread field to model book
#                            thinness more literally.
#   spoof_wash_trading    -- NEW: near-zero jump_probability (genuine price shocks
#                            are not the point of this regime) with a
#                            SignalPredictorSyntheticDiffusionEngine.inject_spoofing_overlay()
#                            pass applied on top -- large, reverting OFI/volume/VPIN
#                            spikes with NO price follow-through, the literature-
#                            grounded proxy for spoofing/wash-trading signatures
#                            (Cartea et al. 2020/2023; Do & Putninš 2023 -- see
#                            inject_spoofing_overlay()'s own docstring). This is the
#                            correctly re-expressed replacement for the legacy
#                            swarm/rl_mechanics.py AdversarialGANSpoofer, which
#                            operated on a raw 8-channel depth representation the
#                            active TradeJackLOBEnv doesn't use.
#   spoof_kwargs=None means no overlay applied -- the regime is exactly the base
#   generator's output. A dict is passed straight through as
#   inject_spoofing_overlay(data_dict, **spoof_kwargs).
DEFAULT_REGIMES: List[Tuple[int, str, float, float, float, Optional[dict]]] = [
    (1, "flash_crash", 6.0, -1.0, 0.015, None),
    (2, "melt_up_squeeze", 6.0, 1.0, 0.015, None),
    (3, "two_sided_high_vol", 4.0, 0.0, 0.02, None),
    (4, "liquidity_vacuum", 10.0, -1.0, 0.004, None),
    (5, "spoof_wash_trading", 1.0, -1.0, 0.001, {"spoof_probability": 0.03, "spoof_duration_ticks": 5}),
]


def inject_synthetic_scenarios(
    symbol: str,
    data_store_dir: Optional[str] = None,
    regimes: Optional[List[Tuple[int, str, float, float, float, Optional[dict]]]] = None,
    num_ticks: int = 2000,
    base_price: float = 65000.0,
    force: bool = False,
) -> List[str]:
    """
    Writes each regime in `regimes` (default DEFAULT_REGIMES) as a physics-schema
    Parquet file at data_store/processed/{symbol}/physics/2099/01/{DD}/physics.parquet.

    Idempotent by default (force=False): skips a regime whose output file already
    exists rather than regenerating it. This is deliberate, not laziness — these are
    meant to be a stable, repeatable stress-test fixture an agent sees roughly the
    same way cycle over cycle, not something that silently reshuffles every time a
    tournament is constructed. Pass force=True to explicitly regenerate (e.g. after
    changing DEFAULT_REGIMES or SyntheticDiffusionEngine's generation logic).

    A regime whose 6th tuple field (spoof_kwargs) is not None gets
    SyntheticDiffusionEngine.inject_spoofing_overlay() applied to the base scenario
    before writing — its own independent RNG stream (see that method's docstring)
    is freshly created per call here, deliberately not seeded from anything
    correlated with the base scenario's generation, for the same
    no-spurious-correlation reason that method exists.

    Returns the list of output paths that are present after this call (whether just
    written or already existing).
    """
    store_dir = data_store_dir or config.data_store_dir
    engine = SyntheticDiffusionEngine(symbol=symbol)
    present: List[str] = []

    for day, name, vol_mult, jump_dir, jump_prob, spoof_kwargs in (regimes or DEFAULT_REGIMES):
        date_path = f"{RESERVED_SYNTHETIC_YEAR}/01/{day:02d}"
        out_dir = os.path.join(store_dir, "processed", symbol, "physics", date_path)
        out_path = os.path.join(out_dir, "physics.parquet")

        if os.path.exists(out_path) and not force:
            logger.debug(f"Synthetic scenario '{name}' already present at {out_path} -- skipping (force=False).")
            present.append(out_path)
            continue

        data_dict = engine._generate_scenario_arrays(base_price, num_ticks, vol_mult, jump_dir, jump_prob)
        if spoof_kwargs is not None:
            data_dict = engine.inject_spoofing_overlay(data_dict, **spoof_kwargs)

        try:
            import polars as pl

            os.makedirs(out_dir, exist_ok=True)
            df = pl.DataFrame(data_dict)
            tmp_path = out_path + ".tmp"
            df.write_parquet(
                tmp_path,
                compression=config.compression_codec,
                compression_level=config.compression_level,
                row_group_size=config.row_group_size,
            )
            os.replace(tmp_path, out_path)
            logger.info(
                f"Injected synthetic scenario '{name}' (dir={jump_dir:+.0f}, "
                f"vol_mult={vol_mult}, jump_prob={jump_prob}, spoof={spoof_kwargs is not None}) -> {out_path}"
            )
            present.append(out_path)
        except ImportError:
            logger.error("Polars required to inject synthetic scenarios -- skipped for this call.")

    return present


def remove_synthetic_scenarios(symbol: str, data_store_dir: Optional[str] = None) -> None:
    """
    Removes the entire reserved synthetic block for `symbol` -- everything under
    data_store/processed/{symbol}/physics/{RESERVED_SYNTHETIC_YEAR}/. Provided
    because the reserved-date design is only safe if it's also trivially
    reversible: an operator who wants a tournament run without synthetic scenarios
    (e.g. to isolate a regression) should be able to remove exactly this block and
    nothing else, without hand-deleting individual day directories.
    """
    import shutil

    store_dir = data_store_dir or config.data_store_dir
    target = os.path.join(store_dir, "processed", symbol, "physics", str(RESERVED_SYNTHETIC_YEAR))
    if os.path.isdir(target):
        shutil.rmtree(target)
        logger.info(f"Removed synthetic scenario block: {target}")
    else:
        logger.info(f"No synthetic scenario block present at {target} -- nothing to remove.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Inject or remove synthetic black-swan scenarios for a symbol.")
    parser.add_argument("--symbol", default="BTC-USDT")
    parser.add_argument("--remove", action="store_true", help="Remove the reserved synthetic block instead of injecting.")
    parser.add_argument("--force", action="store_true", help="Regenerate even if already present.")
    args = parser.parse_args()

    if args.remove:
        remove_synthetic_scenarios(args.symbol)
    else:
        paths = inject_synthetic_scenarios(args.symbol, force=args.force)
        for p in paths:
            print(p)
