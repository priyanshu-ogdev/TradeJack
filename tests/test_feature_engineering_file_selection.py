"""
Executable test for TradeFlowPhysics._select_aggtrades_file. That method has
zero polars/pydantic dependency (pure string/list logic), but the module it
lives in imports data_forge.config at module level, which requires
pydantic_settings -- unavailable in this sandbox, same as every other
data_forge import chain throughout this project. Rather than skip entirely,
this tests a byte-for-byte duplicate of the method body (kept in sync
manually -- see the comment on _select_aggtrades_file_reference below), which
is a real, if imperfect, substitute for actually importing and calling the
live method. The primary test class below still attempts the real import
first and uses it when available (e.g. once pydantic_settings is installed);
the reference-copy class always runs, so this file never silently tests
nothing.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from data_forge.feature_engineering import TradeFlowPhysics
    IMPORT_OK = True
except Exception:
    IMPORT_OK = False


def _select_aggtrades_file_reference(files: list) -> str:
    """Byte-for-byte copy of TradeFlowPhysics._select_aggtrades_file's body
    (see data_forge/feature_engineering.py) -- exists ONLY so this logic has
    at least one real execution path in an environment where the actual
    module can't be imported (no pydantic_settings here). If you change the
    real method, change this copy in the same commit, or this test proves
    nothing about the real code."""
    non_live = [f for f in files if "-live-" not in os.path.basename(f)]
    return non_live[0] if non_live else files[0]


class TestAggTradesFileSelectionReference(unittest.TestCase):
    """Always runs, regardless of whether data_forge can be imported here --
    see module docstring for why this is a reference copy, not the real
    method, and what that limitation means."""

    def test_prefers_bulk_over_live_regardless_of_input_order(self):
        bulk = "/data/raw/BTC-USDT/aggTrades/2026/01/01/BTCUSDT-aggTrades-2026-01-01.parquet"
        live = "/data/raw/BTC-USDT/aggTrades/2026/01/01/BTC-USDT-live-aggTrades-2026-01-01.parquet"
        self.assertEqual(_select_aggtrades_file_reference([bulk, live]), bulk)
        self.assertEqual(_select_aggtrades_file_reference([live, bulk]), bulk)

    def test_falls_back_to_live_when_bulk_absent(self):
        live = "/data/raw/BTC-USDT/aggTrades/2026/01/02/BTC-USDT-live-aggTrades-2026-01-02.parquet"
        self.assertEqual(_select_aggtrades_file_reference([live]), live)

    def test_result_is_order_independent_across_many_permutations(self):
        import itertools
        bulk = "/data/raw/ETH-USDT/aggTrades/2026/02/01/ETHUSDT-aggTrades-2026-02-01.parquet"
        live1 = "/data/raw/ETH-USDT/aggTrades/2026/02/01/ETH-USDT-live-aggTrades-2026-02-01.parquet"
        live2 = "/data/raw/ETH-USDT/aggTrades/2026/02/01/ETH-USDT-live-aggTrades-2026-02-01-2.parquet"
        for perm in itertools.permutations([bulk, live1, live2]):
            self.assertEqual(_select_aggtrades_file_reference(list(perm)), bulk)


@unittest.skipUnless(IMPORT_OK, "data_forge.feature_engineering could not be imported in this environment")
class TestAggTradesFileSelection(unittest.TestCase):
    """Proves the bulk-vs-live selection is deterministic and prefers bulk,
    regardless of the order the (OS-dependent, non-alphabetical) glob would
    have returned them in -- the exact property a prior claimed fix asserted
    but which was not actually present in the shipped file."""

    def test_prefers_bulk_over_live_regardless_of_input_order(self):
        bulk = "/data/raw/BTC-USDT/aggTrades/2026/01/01/BTCUSDT-aggTrades-2026-01-01.parquet"
        live = "/data/raw/BTC-USDT/aggTrades/2026/01/01/BTC-USDT-live-aggTrades-2026-01-01.parquet"

        self.assertEqual(TradeFlowPhysics._select_aggtrades_file([bulk, live]), bulk)
        self.assertEqual(TradeFlowPhysics._select_aggtrades_file([live, bulk]), bulk)

    def test_falls_back_to_live_when_bulk_absent(self):
        live = "/data/raw/BTC-USDT/aggTrades/2026/01/02/BTC-USDT-live-aggTrades-2026-01-02.parquet"
        self.assertEqual(TradeFlowPhysics._select_aggtrades_file([live]), live)

    def test_picks_bulk_deterministically_when_only_bulk_present(self):
        bulk = "/data/raw/BTC-USDT/aggTrades/2026/01/03/BTCUSDT-aggTrades-2026-01-03.parquet"
        self.assertEqual(TradeFlowPhysics._select_aggtrades_file([bulk]), bulk)

    def test_multiple_live_files_same_day_picks_first_of_sorted_input(self):
        """Not this method's job to sort -- process_daily_file() sorts before
        calling it. This only proves _select_aggtrades_file doesn't itself
        reorder or otherwise disturb an already-sorted, all-live input."""
        live_a = "/data/x/BTC-USDT-live-aggTrades-2026-01-04.parquet"
        live_b = "/data/x/BTC-USDT-live-aggTrades-2026-01-04-dup.parquet"
        self.assertEqual(TradeFlowPhysics._select_aggtrades_file([live_a, live_b]), live_a)

    def test_result_is_order_independent_across_many_permutations(self):
        """The property that actually matters: for ANY ordering glob.glob
        might have returned (OS-dependent, not alphabetical), the selection
        must land on the same file. Checks every permutation of a 3-item
        mixed list rather than just the two orderings above."""
        import itertools
        bulk = "/data/raw/ETH-USDT/aggTrades/2026/02/01/ETHUSDT-aggTrades-2026-02-01.parquet"
        live1 = "/data/raw/ETH-USDT/aggTrades/2026/02/01/ETH-USDT-live-aggTrades-2026-02-01.parquet"
        live2 = "/data/raw/ETH-USDT/aggTrades/2026/02/01/ETH-USDT-live-aggTrades-2026-02-01-2.parquet"
        for perm in itertools.permutations([bulk, live1, live2]):
            self.assertEqual(TradeFlowPhysics._select_aggtrades_file(list(perm)), bulk)


if __name__ == "__main__":
    unittest.main(verbosity=2)
