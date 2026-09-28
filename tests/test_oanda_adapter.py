import sys
import os
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from execution.oanda_adapter import to_oanda_instrument, from_oanda_instrument


class TestSymbolConversion(unittest.TestCase):
    def test_no_separator_convention_converts_correctly(self):
        self.assertEqual(to_oanda_instrument("EURUSD"), "EUR_USD")
        self.assertEqual(to_oanda_instrument("USDJPY"), "USD_JPY")

    def test_idempotent_across_separator_styles(self):
        self.assertEqual(to_oanda_instrument("EUR-USD"), "EUR_USD")
        self.assertEqual(to_oanda_instrument("EUR_USD"), "EUR_USD")
        self.assertEqual(to_oanda_instrument("EUR/USD"), "EUR_USD")

    def test_case_insensitive(self):
        self.assertEqual(to_oanda_instrument("eurusd"), "EUR_USD")

    def test_round_trip(self):
        for pair in ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD"]:
            self.assertEqual(from_oanda_instrument(to_oanda_instrument(pair)), pair)

    def test_rejects_non_6_letter_symbol(self):
        with self.assertRaises(ValueError):
            to_oanda_instrument("BTC-USDT")  # a crypto pair, not FX -- must not silently mangle it
        with self.assertRaises(ValueError):
            to_oanda_instrument("EU")

    def test_from_oanda_instrument_strips_underscore(self):
        self.assertEqual(from_oanda_instrument("EUR_USD"), "EURUSD")


if __name__ == "__main__":
    unittest.main(verbosity=2)
