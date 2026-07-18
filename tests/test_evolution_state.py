"""
Verification Test 4: Evolutionary Swarm State & Rollback (`test_evolution_state.py`).
Tests SelfModEngine architecture swap under memory rules, EWC Fisher matrix computation, and GitFinancialRollback.
"""

import os
import sys
import unittest
import shutil
import numpy as np

from swarm.self_mod_manager import SelfModEngine
from swarm.ewc_optimizer import ElasticWeightConsolidation
from swarm.git_rollback import GitFinancialRollback
from swarm.rl_mechanics import HindsightExperienceReplay, PopulationBasedTrainingEngine, AdversarialGANSpoofer


class TestEvolutionAndRollback(unittest.TestCase):

    def setUp(self):
        self.test_state = os.path.abspath("d:/TradeJack/state_test_evo")
        os.makedirs(self.test_state, exist_ok=True)

    def tearDown(self):
        if os.path.exists(self.test_state):
            shutil.rmtree(self.test_state, ignore_errors=True)

    def test_self_mod_engine_tier_rules(self):
        engine = SelfModEngine(child_id=201, state_dir=self.test_state)
        self.assertEqual(engine.active_model.model_name, "Dilated-CNN-Seq2seq")
        
        engine.swap_active_architecture("Attention-is-all-you-Need", current_tier=3)
        self.assertEqual(engine.active_model.model_name, "Deep-Q-learning")
        
        engine.swap_active_architecture("Attention-is-all-you-Need", current_tier=2)
        self.assertEqual(engine.active_model.model_name, "Dilated-CNN-Seq2seq")

    def test_ewc_optimizer_fisher_computation(self):
        engine = SelfModEngine(child_id=202, state_dir=self.test_state)
        mock_data = [(np.random.normal(0, 1, (8, 30, 8)).astype(np.float32), np.ones(8, dtype=np.float32)) for _ in range(2)]
        ewc = ElasticWeightConsolidation(engine.active_model, mock_data)
        self.assertIsNotNone(ewc.fisher_matrix)

    def test_git_financial_rollback(self):
        rollback = GitFinancialRollback(child_id=203, repo_dir=self.test_state, tag_step_dollar=1.0)
        tag = rollback.check_and_checkpoint(current_equity=11.50)
        self.assertIsNotNone(tag)
        self.assertEqual(rollback.hwm_equity, 11.50)
        
        drawdown = (11.50 - 9.50) / 11.50
        did_rollback = rollback.execute_rollback_if_breached(current_equity=9.50, current_drawdown=drawdown)
        self.assertTrue(did_rollback)
        self.assertEqual(rollback.hwm_equity, 11.50)

    def test_rl_mechanics_her_and_pbt(self):
        her = HindsightExperienceReplay(capacity=100)
        her.push(np.zeros(8), [0.5], -0.1, np.ones(8), achieved_equity=9.5, desired_equity=11.0, done=False)
        her.push(np.zeros(8), [0.2], 1.0, np.ones(8), achieved_equity=11.2, desired_equity=11.0, done=True)
        batch = her.sample_with_her(batch_size=2)
        self.assertEqual(len(batch), 2)
        
        pbt = PopulationBasedTrainingEngine(swarm_size=5, exploit_fraction=0.4)
        status = [
            {"child_id": 1, "equity": 15.0, "sharpe_ratio": 2.5, "learning_rate": 0.001},
            {"child_id": 2, "equity": 8.0,  "sharpe_ratio": -0.5, "learning_rate": 0.001}
        ]
        res = pbt.execute_pbt_step(status)
        bottom = next(item for item in res if item["child_id"] == 2)
        self.assertNotEqual(bottom["learning_rate"], 0.001)

    def test_self_mod_safety_guardrails(self):
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

    def test_upgraded_neural_templates(self):
        from swarm.self_mod_manager import AttentionIsAllYouNeedTemplate, DilatedCNNSeq2SeqTemplate, DeepQLearningTemplate
        transformer = AttentionIsAllYouNeedTemplate(input_dim=8, d_model=32, nhead=4, num_layers=2)
        cnn = DilatedCNNSeq2SeqTemplate(input_dim=8, channels=16)
        dqn = DeepQLearningTemplate(input_dim=8, hidden_dim=16)
        
        # Test forward pass with numpy/simulation data
        sample_seq = np.random.normal(0, 1, (8, 20)).astype(np.float32)
        out_t = transformer.forward(sample_seq)
        out_c = cnn.forward(sample_seq)
        out_q = dqn.forward(sample_seq)
        
        self.assertIsNotNone(out_t)
        self.assertIsNotNone(out_c)
        self.assertIsNotNone(out_q)
        
        # Test Dueling Q-Network novelty mutation
        dqn.mutate_novelty(mutation_rate=0.1)


if __name__ == "__main__":
    unittest.main()
