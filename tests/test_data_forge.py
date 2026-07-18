"""
Verification Test 2: Data Forge Subsystems (`test_data_forge.py`).
Tests KvikIO GPUDirect streaming, Polars/NPZ cleaning pipeline, and double-buffer pre-fetching.
"""

import os
import sys
import asyncio
import unittest
import shutil

from data_forge.kvikio_pipeline import KvikIODataForge
from data_forge.parquet_ingest import ParquetIngestPipeline
from data_forge.dali_loader import create_lob_dataloader


class TestDataForgeSubsystems(unittest.TestCase):

    def setUp(self):
        self.test_data_store = os.path.abspath("d:/TradeJack/data_store_test_forge")
        os.makedirs(self.test_data_store, exist_ok=True)

    def tearDown(self):
        if os.path.exists(self.test_data_store):
            shutil.rmtree(self.test_data_store, ignore_errors=True)

    def test_synthetic_generation_and_ingestion(self):
        ingest = ParquetIngestPipeline(data_store_dir=self.test_data_store)
        asyncio.run(ingest.generate_synthetic_crucible_data(symbol="ETH-USDT", num_days=1, ticks_per_day=50))
        
        eth_dir = os.path.join(self.test_data_store, "ETH-USDT")
        self.assertTrue(os.path.exists(eth_dir))
        files = os.listdir(eth_dir)
        self.assertGreater(len(files), 0)

    def test_kvikio_data_forge_streaming(self):
        ingest = ParquetIngestPipeline(data_store_dir=self.test_data_store)
        asyncio.run(ingest.generate_synthetic_crucible_data(symbol="SOL-USDT", num_days=1, ticks_per_day=30))
        
        forge = KvikIODataForge(data_store_dir=self.test_data_store)
        files = forge.scan_available_partitions(symbol="SOL-USDT")
        self.assertGreater(len(files), 0)
        
        tensor_data = forge.load_file_to_tensor(files[0])
        self.assertIsNotNone(tensor_data)
        for col in tensor_data:
            self.assertGreaterEqual(len(tensor_data[col]), 1)
            break

    def test_dali_loader_batch_iteration(self):
        ingest = ParquetIngestPipeline(data_store_dir=self.test_data_store)
        asyncio.run(ingest.generate_synthetic_crucible_data(symbol="ADA-USDT", num_days=1, ticks_per_day=40, start_date="2024-01-01"))
        
        loader = create_lob_dataloader(
            symbol="ADA-USDT", start_date="2024-01-01", end_date="2024-01-01",
            batch_size=8, seq_len=10, data_store_dir=self.test_data_store
        )
        for batch_x, batch_y in loader:
            self.assertIsNotNone(batch_x)
            self.assertEqual(batch_x.shape[0], 8)
            self.assertEqual(batch_x.shape[1], 10)
            self.assertEqual(batch_x.shape[2], 8)
            break


if __name__ == "__main__":
    unittest.main()
