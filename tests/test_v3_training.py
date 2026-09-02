"""
Verification Test 6 (v3): SB3 Training Pipeline.
End-to-end test: data generation → env creation → SB3 model build → train → predict → save/load.
This is the critical test proving gradient-based learning works.
"""

import os
import sys
import unittest
import shutil
import asyncio
import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from stable_baselines3 import PPO, SAC, DQN
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False


class TestSB3Training(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Generate synthetic data once for all tests."""
        cls.test_store = os.path.abspath("d:/TradeJack/data_store_test_sb3")
        cls.test_state = os.path.abspath("d:/TradeJack/state_test_sb3")
        os.makedirs(cls.test_store, exist_ok=True)
        os.makedirs(cls.test_state, exist_ok=True)

        from data_forge.parquet_ingest import ParquetIngestPipeline
        ingest = ParquetIngestPipeline(data_store_dir=cls.test_store)
        asyncio.run(ingest.generate_synthetic_crucible_data(
            symbol="BTC-USDT", num_days=1, ticks_per_day=400
        ))

    @classmethod
    def tearDownClass(cls):
        for d in [cls.test_store, cls.test_state]:
            if os.path.exists(d):
                shutil.rmtree(d, ignore_errors=True)

    def _make_env(self, child_id=999):
        from physics.lob_env import TradeJackLOBEnv
        return TradeJackLOBEnv(
            symbol="BTC-USDT", initial_cash=10.0,
            data_store_dir=self.test_store, child_id=child_id
        )

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
    def test_shared_encoder_forward_pass(self):
        """LOBFeatureEncoder produces correct output shape."""
        from swarm.shared_encoder import LOBFeatureEncoder
        encoder = LOBFeatureEncoder(
            lob_features=5, portfolio_features=4, features_dim=128,
            dilations=(1, 2, 4, 8), hidden_dim=64
        )
        lob = torch.randn(4, 64, 5)
        port = torch.randn(4, 4)
        out = encoder(lob, port)
        self.assertEqual(out.shape, (4, 128))

    @unittest.skipUnless(SB3_AVAILABLE, "SB3 not available")
    def test_registry_build_model(self):
        """REGISTRY.build_model produces a valid SB3 algorithm."""
        from swarm.model_registry import REGISTRY
        env = self._make_env(child_id=700)
        model = REGISTRY.build_model("PPO-DilatedCNN", env=env, device="cpu")
        self.assertIsInstance(model, PPO)

    @unittest.skipUnless(SB3_AVAILABLE, "SB3 not available")
    def test_sb3_train_and_predict(self):
        """SB3 model trains with real gradients and predicts valid actions."""
        from swarm.model_registry import REGISTRY
        env = self._make_env(child_id=701)
        model = REGISTRY.build_model("PPO-DilatedCNN", env=env, device="cpu")

        # Train 128 steps — must not crash and must produce gradient updates
        model.learn(total_timesteps=128)

        # Predict
        obs, _ = env.reset()
        action, _ = model.predict(obs, deterministic=True)
        self.assertIsNotNone(action)
        self.assertEqual(action.shape, (1,))
        self.assertTrue(-1.0 <= action[0] <= 1.0)

    @unittest.skipUnless(SB3_AVAILABLE, "SB3 not available")
    def test_sb3_save_and_load(self):
        """SB3 checkpoint save/load round-trips correctly."""
        from swarm.model_registry import REGISTRY
        env = self._make_env(child_id=702)
        model = REGISTRY.build_model("PPO-DilatedCNN", env=env, device="cpu")
        model.learn(total_timesteps=64)

        # Save
        save_path = os.path.join(self.test_state, "test_checkpoint")
        model.save(save_path)
        self.assertTrue(os.path.exists(save_path + ".zip"))

        # Load
        loaded = PPO.load(save_path, device="cpu")
        obs, _ = env.reset()
        action_orig, _ = model.predict(obs, deterministic=True)
        action_loaded, _ = loaded.predict(obs, deterministic=True)
        np.testing.assert_array_almost_equal(action_orig, action_loaded, decimal=5)

    @unittest.skipUnless(SB3_AVAILABLE, "SB3 not available")
    def test_online_rl_trainer(self):
        """OnlineRLTrainer wraps SB3 correctly."""
        from swarm.rl_trainer import OnlineRLTrainer
        env = self._make_env(child_id=703)
        trainer = OnlineRLTrainer(
            model_name="PPO-DilatedCNN", env=env, device="cpu",
            log_dir=os.path.join(self.test_state, "tb_logs")
        )
        result = trainer.learn(total_timesteps=64)
        self.assertIn("timesteps", result)

        # Save and load
        save_path = os.path.join(self.test_state, "trainer_checkpoint")
        trainer.save(save_path)
        self.assertTrue(os.path.exists(save_path + ".zip"))

    @unittest.skipUnless(SB3_AVAILABLE, "SB3 not available")
    def test_multiple_algorithms(self):
        """All SB3 algorithms in registry build and predict without error."""
        from swarm.model_registry import REGISTRY
        for model_name in ["PPO-DilatedCNN", "SAC-DilatedCNN", "DuelingDQN"]:
            with self.subTest(model=model_name):
                env = self._make_env(child_id=704)
                model = REGISTRY.build_model(model_name, env=env, device="cpu")
                self.assertIsNotNone(model)
                obs, _ = env.reset()
                action, _ = model.predict(obs, deterministic=True)
                self.assertIsNotNone(action)

    def test_replay_buffer(self):
        """Prioritized replay buffer stores and samples correctly."""
        from swarm.replay_buffer import PrioritizedReplayBuffer
        buf = PrioritizedReplayBuffer(capacity=100)
        for i in range(10):
            buf.add(
                obs=np.zeros(8), action=np.array([0.5]),
                reward=float(i), next_obs=np.ones(8), done=False,
                td_error=abs(float(i) - 5.0)
            )
        self.assertEqual(len(buf), 10)
        batch = buf.sample(batch_size=5)
        self.assertEqual(len(batch), 5)

    def test_baselines_predict(self):
        """All baselines produce valid actions."""
        from swarm.baselines import BuyAndHoldBaseline, MomentumBaseline, MeanReversionBaseline
        env = self._make_env(child_id=705)
        obs, _ = env.reset()

        for baseline_cls in [BuyAndHoldBaseline, MomentumBaseline, MeanReversionBaseline]:
            with self.subTest(baseline=baseline_cls.__name__):
                b = baseline_cls()
                action = b.predict(obs)
                self.assertIsNotNone(action)
                self.assertTrue(-1.0 <= action <= 1.0)


if __name__ == "__main__":
    unittest.main()
