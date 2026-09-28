"""
Executable test for SyntheticDiffusionEngine._generate_scenario_arrays -- pure
numpy, zero real dependency on polars/pydantic_settings, but the module it
lives in imports data_forge.config at module level, which needs
pydantic_settings (unavailable in this sandbox, same standing limitation as
everywhere else in data_forge). Uses the same reference-copy pattern as
test_feature_engineering_file_selection.py: a byte-for-byte duplicate of the
method body, kept in sync manually, so this logic has at least one real
execution path here.
"""

import sys
import os
import unittest

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def _generate_scenario_arrays_reference(
    base_price: float, num_ticks: int, volatility_multiplier: float,
    jump_direction: float = -1.0, jump_probability: float = 0.01,
) -> dict:
    """Byte-for-byte copy of SyntheticDiffusionEngine._generate_scenario_arrays's
    body. If you change the real method, change this copy in the same commit,
    or this test proves nothing about the real code."""
    timestamps = np.arange(num_ticks, dtype=np.float64)
    standard_vol = 0.0005
    brownian_shocks = np.random.normal(0, standard_vol, num_ticks)

    jump_probabilities = np.random.uniform(0, 1, num_ticks)
    is_jump = jump_probabilities < jump_probability
    if jump_direction == 0.0:
        jump_sign = np.random.choice([-1.0, 1.0], size=num_ticks)
    else:
        jump_sign = np.full(num_ticks, 1.0 if jump_direction > 0 else -1.0)
    jump_shocks = np.where(
        is_jump,
        jump_sign * np.abs(np.random.normal(0.015, 0.005, num_ticks)) * volatility_multiplier,
        0.0,
    )

    total_shocks = brownian_shocks + jump_shocks
    price_path = base_price * np.exp(np.cumsum(total_shocks))

    ofi = np.random.normal(0, 10, num_ticks)
    ofi += (total_shocks * 10000)

    vpin = np.random.uniform(0.1, 0.3, num_ticks)
    vpin = np.where(is_jump, np.random.uniform(0.7, 0.9, num_ticks), vpin)

    volume = np.random.lognormal(mean=2, sigma=1, size=num_ticks)
    volume = np.where(is_jump, volume * 10, volume)

    kyles_lambda = (total_shocks * price_path) / (ofi + 1e-8)

    data_dict = {
        "timestamp": timestamps,
        "open_price": np.roll(price_path, 1),
        "close_price": price_path,
        "volume": volume,
        "ofi": ofi,
        "vpin_50": vpin,
        "kyles_lambda": kyles_lambda,
    }
    data_dict["open_price"][0] = base_price
    return data_dict


class TestGenerateScenarioArraysReference(unittest.TestCase):
    """Always runs -- see module docstring for the reference-copy caveat."""

    def test_shapes_and_columns(self):
        d = _generate_scenario_arrays_reference(65000.0, 500, 5.0)
        self.assertEqual(set(d.keys()), {"timestamp", "open_price", "close_price", "volume", "ofi", "vpin_50", "kyles_lambda"})
        for col in d.values():
            self.assertEqual(len(col), 500)

    def test_first_open_price_is_base_price_not_a_rolled_artifact(self):
        d = _generate_scenario_arrays_reference(base_price=12345.0, num_ticks=200, volatility_multiplier=5.0)
        self.assertEqual(d["open_price"][0], 12345.0)

    def test_melt_up_squeeze_produces_upward_jumps_not_downward(self):
        """jump_direction=+1.0 must produce a genuinely different regime shape
        (upward jumps), not the same crash pattern relabeled."""
        np.random.seed(7)
        d = _generate_scenario_arrays_reference(65000.0, 2000, 8.0, jump_direction=1.0, jump_probability=0.02)
        log_returns = np.diff(np.log(d["close_price"]))
        big_moves = log_returns[np.abs(log_returns) > 0.02]
        self.assertGreater(len(big_moves), 5, "test setup produced too few large moves to be meaningful")
        self.assertTrue((big_moves > 0).mean() > 0.8, "melt_up_squeeze's large moves should be overwhelmingly positive")

    def test_two_sided_regime_produces_both_directions(self):
        """jump_direction=0.0 must produce jumps in BOTH directions, not
        collapse to one side -- the defining property of a genuinely
        direction-uncertain regime."""
        np.random.seed(11)
        d = _generate_scenario_arrays_reference(65000.0, 3000, 6.0, jump_direction=0.0, jump_probability=0.02)
        log_returns = np.diff(np.log(d["close_price"]))
        big_moves = log_returns[np.abs(log_returns) > 0.02]
        self.assertGreater(len(big_moves), 10, "test setup produced too few large moves to be meaningful")
        up_frac = (big_moves > 0).mean()
        self.assertGreater(up_frac, 0.25)
        self.assertLess(up_frac, 0.75)

    def test_flash_crash_vs_melt_up_are_structurally_different(self):
        """The actual bug this fix addresses: before jump_direction existed,
        every 'regime' was the same one-sided-crash pattern at different
        volatility_multiplier scales. Prove the fix: same seed, same
        volatility_multiplier, opposite jump_direction -> the two resulting
        price paths must trend in opposite directions overall."""
        np.random.seed(99)
        crash = _generate_scenario_arrays_reference(65000.0, 2000, 8.0, jump_direction=-1.0, jump_probability=0.02)
        np.random.seed(99)
        squeeze = _generate_scenario_arrays_reference(65000.0, 2000, 8.0, jump_direction=1.0, jump_probability=0.02)
        crash_total_return = np.log(crash["close_price"][-1] / crash["close_price"][0])
        squeeze_total_return = np.log(squeeze["close_price"][-1] / squeeze["close_price"][0])
        self.assertLess(crash_total_return, 0)
        self.assertGreater(squeeze_total_return, 0)

    def test_liquidity_vacuum_style_params_produce_rare_but_extreme_moves(self):
        """DEFAULT_REGIMES' liquidity_vacuum uses low jump_probability + high
        volatility_multiplier -- confirms that combination actually produces
        the intended shape: few jump ticks, but each one large."""
        np.random.seed(3)
        d = _generate_scenario_arrays_reference(65000.0, 5000, 10.0, jump_direction=-1.0, jump_probability=0.004)
        log_returns = np.diff(np.log(d["close_price"]))
        big_moves = log_returns[np.abs(log_returns) > 0.03]
        self.assertLess(len(big_moves) / len(log_returns), 0.01)
        self.assertGreater(len(big_moves), 0, "test setup produced zero large moves -- liquidity_vacuum params may not actually be extreme enough")


def _inject_spoofing_overlay_reference(
    data_dict: dict, spoof_probability: float = 0.02, spoof_duration_ticks: int = 5,
    ofi_multiplier: float = 50.0, volume_multiplier: float = 10.0, vpin_range: tuple = (0.7, 0.9),
    rng=None,
) -> dict:
    """Byte-for-byte copy of SyntheticDiffusionEngine.inject_spoofing_overlay's body."""
    rng = rng or np.random.default_rng()
    out = {k: np.array(v, copy=True) for k, v in data_dict.items()}
    num_ticks = len(out["close_price"])
    onset_draws = rng.uniform(0, 1, num_ticks)
    is_onset = onset_draws < spoof_probability
    t = 0
    while t < num_ticks:
        if is_onset[t]:
            end = min(t + spoof_duration_ticks, num_ticks)
            out["ofi"][t:end] = out["ofi"][t:end] * ofi_multiplier
            out["volume"][t:end] = out["volume"][t:end] * volume_multiplier
            out["vpin_50"][t:end] = rng.uniform(vpin_range[0], vpin_range[1], end - t)
            t = end
        else:
            t += 1
    return out


class TestSpoofingOverlayReference(unittest.TestCase):
    """Proves the spoof overlay is genuinely decoupled from price and genuinely
    reverts -- the two properties the whole mechanism's economic validity
    depends on (see docs/PHASE3_SCENARIO_DIVERSITY.md's research section)."""

    def test_price_and_kyles_lambda_completely_unchanged(self):
        base = _generate_scenario_arrays_reference(65000.0, 2000, 5.0)
        spoofed = _inject_spoofing_overlay_reference(base, spoof_probability=0.05, rng=np.random.default_rng(1))
        np.testing.assert_array_equal(base["close_price"], spoofed["close_price"])
        np.testing.assert_array_equal(base["open_price"], spoofed["open_price"])
        np.testing.assert_array_equal(base["kyles_lambda"], spoofed["kyles_lambda"])

    def test_ofi_volume_vpin_genuinely_elevated_during_spoof_windows(self):
        base = _generate_scenario_arrays_reference(65000.0, 5000, 1.0)  # low vol baseline
        rng = np.random.default_rng(2)
        spoofed = _inject_spoofing_overlay_reference(base, spoof_probability=0.03, rng=rng)
        changed = spoofed["ofi"] != base["ofi"]
        self.assertGreater(changed.sum(), 10, "test setup produced too few spoof ticks to be meaningful")
        self.assertGreater(
            np.abs(spoofed["ofi"][changed]).mean(), np.abs(base["ofi"]).mean() * 5,
            "spoofed OFI ticks should be dramatically elevated vs. baseline",
        )
        self.assertGreater(spoofed["volume"][changed].mean(), base["volume"].mean() * 5)
        self.assertTrue((spoofed["vpin_50"][changed] >= 0.7).all())

    def test_decoupled_from_price_jumps_not_merely_also_random(self):
        """The actual claim under test: spoof windows and real price-jump ticks
        are statistically independent. Run many seeds and confirm spoof-window
        ticks are not disproportionately jump ticks (which would indicate a
        hidden correlation, e.g. from accidentally sharing RNG state)."""
        overlap_fractions = []
        for seed in range(30):
            np.random.seed(seed)  # controls the jump-diffusion draws (global RNG)
            base = _generate_scenario_arrays_reference(65000.0, 3000, 5.0, jump_probability=0.02)
            log_returns = np.abs(np.diff(np.log(base["close_price"]), prepend=np.log(65000.0)))
            is_real_jump = log_returns > np.percentile(log_returns, 98)  # top 2% moves, proxy for jump ticks

            spoofed = _inject_spoofing_overlay_reference(
                base, spoof_probability=0.02, rng=np.random.default_rng(seed + 1000)  # independent stream
            )
            is_spoofed = spoofed["ofi"] != base["ofi"]
            if is_spoofed.sum() == 0:
                continue
            overlap_fractions.append(is_real_jump[is_spoofed].mean())

        # If genuinely decoupled, spoofed ticks should land on "real jump" ticks
        # at roughly the base rate (~2%), not systematically more or less.
        mean_overlap = np.mean(overlap_fractions)
        self.assertLess(mean_overlap, 0.15, f"spoof windows overlap real jumps far more than chance ({mean_overlap:.3f}) -- suspect shared RNG state")

    def test_reversion_is_exact_not_gradual_decay(self):
        """At exactly spoof_duration_ticks after onset, values must snap back to
        their PRE-SPOOF baseline -- not decay toward it, and not stay elevated."""
        base = _generate_scenario_arrays_reference(65000.0, 200, 1.0)
        rng = np.random.default_rng(42)
        # Force a single, known spoof window by directly testing the overlay's
        # onset-detection against a controlled uniform draw via monkeypatched rng.
        class _FixedRng:
            def uniform(self, lo, hi, size=None):
                if size == 200:  # the onset draw call
                    arr = np.ones(200)
                    arr[50] = 0.0  # force onset exactly at t=50
                    return arr
                return np.full(size, (lo + hi) / 2.0)  # vpin fill call
        spoofed = _inject_spoofing_overlay_reference(base, spoof_probability=0.01, spoof_duration_ticks=5, rng=_FixedRng())
        # Ticks 50-54 (5 ticks) are the spoof window; tick 55 must be back to baseline.
        for i in range(50, 55):
            self.assertNotEqual(spoofed["ofi"][i], base["ofi"][i])
        self.assertEqual(spoofed["ofi"][55], base["ofi"][55])
        self.assertEqual(spoofed["volume"][55], base["volume"][55])

    def test_does_not_mutate_input_dict(self):
        base = _generate_scenario_arrays_reference(65000.0, 500, 5.0)
        base_ofi_copy = base["ofi"].copy()
        _inject_spoofing_overlay_reference(base, spoof_probability=0.05, rng=np.random.default_rng(5))
        np.testing.assert_array_equal(base["ofi"], base_ofi_copy)


if __name__ == "__main__":
    unittest.main(verbosity=2)
