"""
DVC Tracker for 3TB Omni-Forge.
Provides snapshot rollback capabilities for the massive Data Store, bypassing Iceberg bloat.
Allows the Warden to checkout specific market regimes (e.g. "bull_run_2024", "flash_crash_2022") for agent retraining.

SOTA Upgrades:
  - list_tracked_snapshots(): enumerate all DVC-tracked tags for the storage manager
  - is_tracked(file_path): check if a file is safely tracked before LRU eviction
"""

import os
import subprocess
import logging

from data_forge.config import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] (DVCTracker) %(message)s")
logger = logging.getLogger("DVCTracker")

class DVCTracker:
    def __init__(self):
        self.store_dir = config.data_store_dir

    def _run_cmd(self, cmd: list, timeout: int = 60) -> tuple:
        """Runs a subprocess command and returns (success, stdout)."""
        try:
            result = subprocess.run(
                cmd, cwd=self.store_dir, capture_output=True, text=True,
                check=True, timeout=timeout
            )
            logger.debug(f"Command {' '.join(cmd)} succeeded: {result.stdout}")
            return True, result.stdout.strip()
        except subprocess.CalledProcessError as e:
            logger.error(f"Command {' '.join(cmd)} failed: {e.stderr}")
            return False, e.stderr.strip()
        except subprocess.TimeoutExpired:
            logger.error(f"Command {' '.join(cmd)} timed out after {timeout}s")
            return False, ""
        except FileNotFoundError:
            logger.error(f"Command '{cmd[0]}' not found. Is it installed and in PATH?")
            return False, ""

    def init_dvc(self):
        """Initializes a DVC repository in the data_store if it doesn't exist."""
        dvc_dir = os.path.join(self.store_dir, ".dvc")
        if not os.path.exists(dvc_dir):
            logger.info("Initializing new DVC repository in data_store...")
            # Ensure it's a git repo first
            if not os.path.exists(os.path.join(self.store_dir, ".git")):
                self._run_cmd(["git", "init"])

            success, _ = self._run_cmd(["dvc", "init"])
            if success:
                logger.info("DVC initialized successfully.")
            else:
                logger.error("Failed to initialize DVC. Is DVC installed?")
        else:
            logger.info("DVC is already initialized.")

    def track_data(self, tag_name: str, description: str = ""):
        """Tracks the current state of the raw and processed directories and creates a snapshot tag."""
        logger.info(f"Tracking current data state as '{tag_name}'...")

        # Track directories
        raw_path = "raw"
        processed_path = "processed"

        self._run_cmd(["dvc", "add", raw_path, processed_path])

        # Commit to Git to save the DVC meta files
        self._run_cmd(["git", "add", f"{raw_path}.dvc", f"{processed_path}.dvc", ".gitignore"])
        self._run_cmd(["git", "commit", "-m", f"Snapshot: {tag_name}. {description}"])

        # Tag the commit
        self._run_cmd(["git", "tag", "-a", tag_name, "-m", description])
        logger.info(f"Successfully tracked data and tagged as '{tag_name}'.")

    def checkout_snapshot(self, tag_name: str):
        """Rolls back the 3TB data store to a specific snapshot (tag)."""
        logger.warning(f"Rolling back Omni-Forge data store to snapshot: {tag_name}")

        # Checkout the git tag containing the .dvc files
        git_success, _ = self._run_cmd(["git", "checkout", tag_name])

        if git_success:
            # Pull the actual data from the DVC cache
            dvc_success, _ = self._run_cmd(["dvc", "checkout"])
            if dvc_success:
                logger.info(f"Successfully rolled back data to {tag_name}.")
            else:
                logger.error("DVC checkout failed.")
        else:
            logger.error(f"Git checkout for tag '{tag_name}' failed.")

    def list_tracked_snapshots(self) -> list:
        """
        Returns a list of all DVC-tracked git tags (snapshot names).
        Used by the StorageManager to verify data is tracked before LRU eviction.
        """
        success, output = self._run_cmd(["git", "tag", "--list"])
        if success and output:
            tags = [t.strip() for t in output.split("\n") if t.strip()]
            logger.debug(f"Found {len(tags)} tracked snapshots.")
            return tags
        return []

    def is_tracked(self, file_path: str) -> bool:
        """
        Checks if a specific file is tracked by DVC (has a corresponding .dvc meta file).
        Used by the StorageManager before evicting files to ensure they can be re-pulled.
        """
        # Check for a direct .dvc tracking file
        dvc_file = file_path + ".dvc"
        if os.path.exists(dvc_file):
            return True

        # Check if the file's parent directory is tracked
        rel_path = os.path.relpath(file_path, self.store_dir)
        parts = rel_path.split(os.sep)

        # Walk up the directory tree looking for .dvc files
        for i in range(len(parts)):
            potential_dvc = os.path.join(self.store_dir, *parts[:i+1]) + ".dvc"
            if os.path.exists(potential_dvc):
                return True

        return False

    def remove_tracking(self, file_path: str) -> bool:
        """
        Removes DVC tracking for a specific file (called by StorageManager during eviction).
        This ensures the DVC graph stays in sync with the physical disk state.
        """
        dvc_file = file_path + ".dvc"
        if os.path.exists(dvc_file):
            success, _ = self._run_cmd(["dvc", "remove", dvc_file])
            if success:
                logger.info(f"Removed DVC tracking for {file_path}")
                return True
            else:
                logger.error(f"Failed to remove DVC tracking for {file_path}")
                return False
        return True  # File wasn't tracked, nothing to remove


if __name__ == "__main__":
    logger.info("Starting DVC Tracker...")
    tracker = DVCTracker()
    # Test initialization (Requires DVC installed on system)
    # tracker.init_dvc()
    snapshots = tracker.list_tracked_snapshots()
    logger.info(f"Tracked snapshots: {snapshots}")
    logger.info("DVC Tracker Ready for integration with Warden.")
