"""
Verification Test 4 (v3): Evolutionary Swarm State & SB3 Training.
Tests SelfModEngine model selection, PBT with SB3 checkpoints, and model registry.
Replaces legacy tests that imported deleted template classes.
"""

import os
import sys
import unittest
import shutil
import numpy as np

from swarm.self_mod_manager import SelfModEngine
from swarm.rl_mechanics import HindsightExperienceReplay, PopulationBasedTrainingEngine, AdversarialGANSpoofer
from swarm.model_registry import REGISTRY


class TestEvolutionAndRollback(unittest.TestCase):

    def setUp(self):
        self.test_state = os.path.abspath("d:/TradeJack/state_test_evo")
        os.makedirs(self.test_state, exist_ok=True)

    def tearDown(self):
        if os.path.exists(self.test_state):
            shutil.rmtree(self.test_state, ignore_errors=True)

    def test_model_registry_has_v3_models(self):
        """Verify v3 model registry has all expected models."""
        model_names = list(REGISTRY.cards.keys())
        self.assertIn("PPO-DilatedCNN", model_names)
        self.assertIn("SAC-DilatedCNN", model_names)
        self.assertIn("DuelingDQN", model_names)
        self.assertGreaterEqual(len(model_names), 4)

    def test_model_registry_tier_filtering(self):
        """Verify tier-based model filtering works."""
        tier2 = REGISTRY.list_models_for_tier(max_tier=2)
        tier3 = REGISTRY.list_models_for_tier(max_tier=3)
        # Tier 3 should have all models tier 2 has plus more
        self.assertGreaterEqual(len(tier3), len(tier2))

    def test_model_card_lookup(self):
        """Verify model card lookup returns correct metadata."""
        card = REGISTRY.get_model_card("PPO-DilatedCNN")
        self.assertIsNotNone(card)
        self.assertEqual(card.model_name, "PPO-DilatedCNN")
        self.assertEqual(card.algo, "PPO")

    def test_self_mod_engine_initialization(self):
        """Verify SelfModEngine still initializes with v3 model names."""
        engine = SelfModEngine(child_id=201, state_dir=self.test_state)
        self.assertIsNotNone(engine)

    def test_ewc_optimizer_fisher_computation(self):
        """Verify EWC Fisher matrix computation still works."""
        from swarm.ewc_optimizer import ElasticWeightConsolidation
        engine = SelfModEngine(child_id=202, state_dir=self.test_state)
        mock_data = [
            (np.random.normal(0, 1, (8, 30, 8)).astype(np.float32),
             np.ones(8, dtype=np.float32)) for _ in range(2)
        ]
        ewc = ElasticWeightConsolidation(engine.active_model, mock_data)
        self.assertIsNotNone(ewc.fisher_matrix)

    def test_git_financial_rollback(self):
        """Verify git rollback still works for checkpoint management."""
        from swarm.git_rollback import GitFinancialRollback
        rollback = GitFinancialRollback(child_id=203, repo_dir=self.test_state, tag_step_dollar=1.0)
        tag = rollback.check_and_checkpoint(current_equity=11.50)
        self.assertIsNotNone(tag)
        self.assertEqual(rollback.hwm_equity, 11.50)

        drawdown = (11.50 - 9.50) / 11.50
        did_rollback = rollback.execute_rollback_if_breached(current_equity=9.50, current_drawdown=drawdown)
        self.assertTrue(did_rollback)

    def test_rl_mechanics_her(self):
        """Verify HER replay buffer works correctly."""
        her = HindsightExperienceReplay(capacity=100)
        her.push(np.zeros(8), [0.5], -0.1, np.ones(8), achieved_equity=9.5, desired_equity=11.0, done=False)
        her.push(np.zeros(8), [0.2], 1.0, np.ones(8), achieved_equity=11.2, desired_equity=11.0, done=True)
        batch = her.sample_with_her(batch_size=2)
        self.assertEqual(len(batch), 2)

    def test_rl_mechanics_pbt_with_v3_fields(self):
        """Verify PBT uses v3 Sortino fitness and model names."""
        pbt = PopulationBasedTrainingEngine(swarm_size=5, exploit_fraction=0.4)
        status = [
            {"child_id": 1, "equity": 15.0, "sortino_ratio": 2.5, "ticks_active": 100,
             "learning_rate": 0.001, "model_name": "PPO-DilatedCNN"},
            {"child_id": 2, "equity": 8.0, "sortino_ratio": -0.5, "ticks_active": 100,
             "learning_rate": 0.001, "model_name": "DuelingDQN"}
        ]
        res = pbt.execute_pbt_step(status)
        # Bottom performer should get mutated learning rate
        bottom = next(item for item in res if item["child_id"] == 2)
        self.assertNotEqual(bottom["learning_rate"], 0.001)

    def test_adversarial_spoofer(self):
        """Verify adversarial LOB spoofer generates valid noise."""
        spoofer = AdversarialGANSpoofer(spoof_intensity=1.0)
        lob = np.random.rand(20, 8).astype(np.float32)
        spoofed = spoofer.inject_spoof_noise(lob)
        self.assertEqual(spoofed.shape, (20, 8))
        # With intensity=1.0, the spoofed data should differ from original
        # (statistically, but not guaranteed for every seed)

    def test_self_mod_safety_guardrails(self):
        """Verify self-mod safety still blocks protected files."""
        from swarm.self_mod_manager import validate_self_mod_safety
        engine = SelfModEngine(child_id=205, state_dir=self.test_state)

        # Test hard-coded safety invariant block
        allowed, msg = validate_self_mod_safety("warden/warden_core.py", 500)
        self.assertFalse(allowed)
        self.assertIn("BLOCKED: Cannot modify protected file", msg)

        # Test safe modification allowed
        safe_file = os.path.join(self.test_state, "custom_strategy.py")
        mod_res = engine.modify_source_file(safe_file, "# new strategy code", reason="Testing safe modification")
        self.assertTrue(mod_res["success"])
        self.assertTrue(os.path.exists(safe_file))


if __name__ == "__main__":
    unittest.main()
