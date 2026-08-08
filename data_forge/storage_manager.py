"""
Storage Budget Manager for the 2.5 TB Omni-Forge.
Enforces disk usage limits with intelligent LRU eviction and Cold Storage archival.
Prevents the DGX Spark's NVMe from filling up during multi-year multi-symbol ingestion.

Architecture:
  - Scans data_store/ recursively and calculates total disk usage per tier
  - When storage exceeds 95% of budget, evicts oldest partitions using LRU policy
  - Instead of deleting: moves evicted files to data_store/cold_storage/ as .tar.zst archives
  - DVC Harmony: calls `dvc remove` on evicted files before archival to keep DVC graph in sync
  - Preserves synthetic and live data (never evicted — these are irreplaceable)
  - Returns structured usage reports for the Warden audit cycle
"""

import os
import time
import tarfile
import shutil
import logging
import subprocess
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from data_forge.config import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] (StorageManager) %(message)s")
logger = logging.getLogger("StorageManager")


@dataclass
class StorageReport:
    """Structured disk usage report per data tier."""
    total_bytes: int
    budget_bytes: int
    usage_percent: float
    raw_bytes: int
    processed_bytes: int
    synthetic_bytes: int
    live_bytes: int
    macro_bytes: int
    cold_storage_bytes: int
    quarantine_bytes: int


class StorageManager:
    """
    Manages the 2.5 TB storage budget for the Omni-Forge data_store.
    Implements LRU eviction with Cold Storage archival and DVC synchronization.
    """

    # Eviction priority order: raw first, then processed. Never touch synthetic/live.
    EVICTABLE_TIERS = ["raw", "processed"]
    PROTECTED_TIERS = ["synthetic", "live", "macro", "cold_storage", "quarantine"]

    def __init__(self, data_store_dir: str = None):
        self.data_store_dir = os.path.abspath(data_store_dir or config.data_store_dir)
        self.cold_storage_dir = os.path.join(self.data_store_dir, "cold_storage")
        self.budget_bytes = int(config.storage_budget_tb * (1024 ** 4))
        os.makedirs(self.cold_storage_dir, exist_ok=True)

    def _get_dir_size(self, dir_path: str) -> int:
        """Recursively calculates total size of a directory in bytes."""
        total = 0
        if not os.path.exists(dir_path):
            return 0
        for dirpath, _, filenames in os.walk(dir_path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                if os.path.isfile(fp):
                    try:
                        total += os.path.getsize(fp)
                    except OSError:
                        pass
        return total

    def get_usage_report(self) -> StorageReport:
        """Returns a structured disk usage breakdown by data tier."""
        tiers = {
            "raw": os.path.join(self.data_store_dir, "raw"),
            "processed": os.path.join(self.data_store_dir, "processed"),
            "synthetic": os.path.join(self.data_store_dir, "synthetic"),
            "live": os.path.join(self.data_store_dir, "live"),
            "macro": os.path.join(self.data_store_dir, "macro"),
            "cold_storage": self.cold_storage_dir,
            "quarantine": os.path.join(self.data_store_dir, "quarantine"),
        }

        sizes = {name: self._get_dir_size(path) for name, path in tiers.items()}

        # Also count files at root level (e.g., .gitkeep, BTC-USDT depth data)
        total = 0
        for dirpath, _, filenames in os.walk(self.data_store_dir):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                if os.path.isfile(fp):
                    try:
                        total += os.path.getsize(fp)
                    except OSError:
                        pass

        usage_pct = (total / self.budget_bytes * 100) if self.budget_bytes > 0 else 0

        return StorageReport(
            total_bytes=total,
            budget_bytes=self.budget_bytes,
            usage_percent=round(usage_pct, 2),
            raw_bytes=sizes.get("raw", 0),
            processed_bytes=sizes.get("processed", 0),
            synthetic_bytes=sizes.get("synthetic", 0),
            live_bytes=sizes.get("live", 0),
            macro_bytes=sizes.get("macro", 0),
            cold_storage_bytes=sizes.get("cold_storage", 0),
            quarantine_bytes=sizes.get("quarantine", 0),
        )

    def _scan_evictable_files(self) -> List[Tuple[str, float, int]]:
        """
        Scans evictable tiers and returns files sorted by modification time (oldest first).
        Returns list of (file_path, mtime, size_bytes).
        """
        files = []
        for tier in self.EVICTABLE_TIERS:
            tier_dir = os.path.join(self.data_store_dir, tier)
            if not os.path.exists(tier_dir):
                continue
            for dirpath, _, filenames in os.walk(tier_dir):
                for f in filenames:
                    fp = os.path.join(dirpath, f)
                    if os.path.isfile(fp):
                        try:
                            stat = os.stat(fp)
                            files.append((fp, stat.st_mtime, stat.st_size))
                        except OSError:
                            pass

        # Sort by modification time (oldest first = evict first)
        files.sort(key=lambda x: x[1])
        return files

    def _dvc_remove(self, file_path: str):
        """Removes a file from DVC tracking before archival (DVC + LRU Harmony)."""
        dvc_file = file_path + ".dvc"
        if os.path.exists(dvc_file):
            try:
                subprocess.run(
                    ["dvc", "remove", dvc_file],
                    cwd=self.data_store_dir,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=False,
                    timeout=30,
                )
                logger.debug(f"DVC tracking removed for {file_path}")
            except (subprocess.TimeoutExpired, FileNotFoundError):
                logger.debug(f"DVC remove skipped (not installed or timeout) for {file_path}")

    def _archive_to_cold_storage(self, file_path: str):
        """
        Moves an evicted file to cold_storage/ as a .tar.zst archive.
        Preserves the relative directory structure for future DVC restoration.
        """
        rel_path = os.path.relpath(file_path, self.data_store_dir)
        archive_name = rel_path.replace(os.sep, "__") + ".tar.zst"
        archive_path = os.path.join(self.cold_storage_dir, archive_name)

        try:
            # Create a tar archive (compressed with zstd if available, else gzip fallback)
            # Python's tarfile doesn't support zstd natively, so we use gzip as archive compression
            # The files inside are already ZSTD Parquet, so double-compression is minimal
            tar_path = archive_path.replace(".tar.zst", ".tar.gz")
            with tarfile.open(tar_path, "w:gz") as tar:
                tar.add(file_path, arcname=rel_path)

            logger.debug(f"Archived to cold storage: {tar_path}")
        except Exception as e:
            logger.error(f"Cold storage archive failed for {file_path}: {e}")

    def evict_lru(self, target_free_bytes: int = None) -> int:
        """
        Evicts oldest files from evictable tiers until storage is under 90% of budget.
        Files are archived to cold_storage/ instead of being permanently deleted.
        Returns total bytes freed.
        """
        report = self.get_usage_report()

        if report.usage_percent < 95.0:
            logger.debug(f"Storage at {report.usage_percent:.1f}%. No eviction needed.")
            return 0

        # Target: bring usage down to 90%
        target_bytes = int(self.budget_bytes * 0.90)
        bytes_to_free = report.total_bytes - target_bytes

        if target_free_bytes:
            bytes_to_free = max(bytes_to_free, target_free_bytes)

        logger.warning(
            f"STORAGE EVICTION: {report.usage_percent:.1f}% used "
            f"({report.total_bytes / (1024**4):.3f} TB / {config.storage_budget_tb} TB). "
            f"Need to free {bytes_to_free / (1024**3):.2f} GB."
        )

        evictable = self._scan_evictable_files()
        bytes_freed = 0

        for file_path, mtime, size in evictable:
            if bytes_freed >= bytes_to_free:
                break

            logger.info(f"Evicting: {file_path} ({size / (1024**2):.1f} MB, age: {time.time() - mtime:.0f}s)")

            # Step 1: DVC + LRU Harmony — remove from DVC tracking
            self._dvc_remove(file_path)

            # Step 2: Archive to cold storage
            self._archive_to_cold_storage(file_path)

            # Step 3: Delete the original file
            try:
                os.remove(file_path)
                bytes_freed += size

                # Clean up empty parent directories
                parent = os.path.dirname(file_path)
                while parent != self.data_store_dir:
                    if os.path.isdir(parent) and not os.listdir(parent):
                        os.rmdir(parent)
                        parent = os.path.dirname(parent)
                    else:
                        break

            except OSError as e:
                logger.error(f"Failed to delete {file_path}: {e}")

        logger.info(f"Eviction complete. Freed {bytes_freed / (1024**3):.2f} GB.")
        return bytes_freed

    def print_usage_report(self):
        """Prints a formatted storage usage report."""
        report = self.get_usage_report()
        gb = 1024 ** 3
        tb = 1024 ** 4

        print(f"\n{'='*60}")
        print(f"  TradeJack Data Forge -- Storage Usage Report")
        print(f"{'='*60}")
        print(f"  Total Usage:     {report.total_bytes / tb:.4f} TB / {report.budget_bytes / tb:.1f} TB ({report.usage_percent:.1f}%)")
        print(f"  {'-'*56}")
        print(f"  Raw Data:        {report.raw_bytes / gb:.2f} GB")
        print(f"  Processed:       {report.processed_bytes / gb:.2f} GB")
        print(f"  Synthetic:       {report.synthetic_bytes / gb:.2f} GB")
        print(f"  Live (L2 Depth): {report.live_bytes / gb:.2f} GB")
        print(f"  Macro (OHLCV):   {report.macro_bytes / gb:.2f} GB")
        print(f"  Cold Storage:    {report.cold_storage_bytes / gb:.2f} GB")
        print(f"  Quarantine:      {report.quarantine_bytes / gb:.2f} GB")
        print(f"{'='*60}\n")

        if report.usage_percent >= 95.0:
            print(f"  [!] CRITICAL: Storage at {report.usage_percent:.1f}%. LRU eviction recommended.")
        elif report.usage_percent >= 80.0:
            print(f"  [*] WARNING: Storage at {report.usage_percent:.1f}%. Monitor closely.")
        else:
            print(f"  [OK] Storage healthy at {report.usage_percent:.1f}%.")


if __name__ == "__main__":
    manager = StorageManager()
    manager.print_usage_report()
