"""
ContinuousTrainer can't be constructed in this sandbox (its __init__ reaches
PlasticityManager, which imports torch) -- so this tests the exact derivation
logic as a standalone reference copy, same pattern as
test_feature_engineering_file_selection.py and test_synthetic_diffusion.py's
reference-copy tests. If you change the real derivation in
ContinuousTrainer.__init__, change this copy in the same commit.
"""

import os
import unittest


def _derive_deployed_model_path(state_dir: str, deployed_model_path: str | None) -> str:
    """Byte-for-byte copy of the derivation logic in
    ContinuousTrainer.__init__ (training/continuous_trainer.py)."""
    resolved_state_dir = os.path.abspath(state_dir)
    if deployed_model_path:
        return os.path.abspath(deployed_model_path)
    return os.path.join(resolved_state_dir, "deployed", "weights_promoted")


class TestDeployedModelPathDerivation(unittest.TestCase):
    def test_default_derives_from_custom_state_dir(self):
        """The actual bug: overriding state_dir used to leave
        deployed_model_path on its own independent bare-relative default,
        so the two would silently disagree unless launched from a
        particular working directory by coincidence."""
        result = _derive_deployed_model_path("/var/tradejack/state", None)
        self.assertEqual(result, "/var/tradejack/state/deployed/weights_promoted")

    def test_relative_state_dir_still_derives_consistently(self):
        result = _derive_deployed_model_path("state", None)
        self.assertEqual(result, os.path.join(os.path.abspath("state"), "deployed", "weights_promoted"))
        self.assertTrue(os.path.isabs(result), "derived path must always be absolute, even from a relative state_dir")

    def test_explicit_override_still_wins(self):
        """An explicit deployed_model_path must still be respected -- this fix
        only changes what the DEFAULT derives from, not what a caller who
        genuinely wants a decoupled path (e.g. a model store shared across
        multiple state_dirs) can still do."""
        result = _derive_deployed_model_path("/var/tradejack/state", "/shared/models/prod")
        self.assertEqual(result, "/shared/models/prod")

    def test_two_different_state_dirs_never_collide(self):
        """The property that actually matters end to end: two differently-
        configured trainers must never derive the same model path."""
        a = _derive_deployed_model_path("/deploy/a/state", None)
        b = _derive_deployed_model_path("/deploy/b/state", None)
        self.assertNotEqual(a, b)


if __name__ == "__main__":
    unittest.main(verbosity=2)
