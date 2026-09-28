"""
warden/deployment_monitor.py - Live Deployment Monitoring Band
Segment 6: Reads deployed model's realized daily return from ledger.
Compares against empirical band from Crucible results (mean +/- 2 sigma).
Flags for manual review if out-of-band N consecutive days.
Monitoring triggers human review ONLY - never autonomous correction.
"""

import os
import sys
import time
import json
import math
import sqlite3
import logging
from typing import Optional, List, Dict, Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] (DeployMonitor) %(message)s")
logger = logging.getLogger("DeployMonitor")


class DeploymentMonitor:
    """
    Segment 6 - Live Monitoring Band.
    
    Reads daily realized returns from the deployed agent ledger and compares
    them against the empirical Crucible band (mean +/- 2 sigma).
    Flags human review if return stays out-of-band for N consecutive days.
    Never issues autonomous corrections.
    """

    def __init__(
        self,
        deploy_ledger_path: str,
        crucible_stats_path: str = "state/crucible_stats.json",
        out_of_band_threshold: int = 3,
        sigma_multiplier: float = 2.0
    ):
        self.deploy_ledger_path = os.path.abspath(deploy_ledger_path)
        self.crucible_stats_path = os.path.abspath(crucible_stats_path)
        self.out_of_band_threshold = out_of_band_threshold
        self.sigma_multiplier = sigma_multiplier
        self._consecutive_ood_days: int = 0
        self._alert_log: List[Dict[str, Any]] = []

    def load_crucible_band(self) -> Optional[Dict[str, float]]:
        """
        Loads the empirical return band computed from Crucible results.
        File format: {"mean_daily_return": 0.003, "std_daily_return": 0.012}
        Returns None if not yet available.
        """
        if not os.path.exists(self.crucible_stats_path):
            logger.warning(f"Crucible stats not found at {self.crucible_stats_path}. Band monitoring unavailable.")
            return None
        try:
            with open(self.crucible_stats_path, "r") as f:
                stats = json.load(f)
            mean = float(stats.get("mean_daily_return", 0.0))
            std = float(stats.get("std_daily_return", 0.01))
            return {
                "mean": mean,
                "std": std,
                "lower": mean - self.sigma_multiplier * std,
                "upper": mean + self.sigma_multiplier * std,
            }
        except Exception as e:
            logger.error(f"Failed to load crucible stats: {e}")
            return None

    def get_latest_daily_return(self) -> Optional[float]:
        """
        Reads the last two portfolio_state rows to compute the realized daily return.
        Returns None if ledger is unavailable or has < 2 rows.
        """
        if not os.path.exists(self.deploy_ledger_path):
            logger.warning(f"Deploy ledger not found: {self.deploy_ledger_path}")
            return None
        try:
            conn = sqlite3.connect(self.deploy_ledger_path, timeout=5)
            cursor = conn.cursor()
            cursor.execute("SELECT equity FROM portfolio_state ORDER BY tick_id DESC LIMIT 2")
            rows = cursor.fetchall()
            conn.close()
            if len(rows) < 2:
                return None
            current_equity = rows[0][0]
            prior_equity = rows[1][0]
            if prior_equity <= 0:
                return None
            return (current_equity - prior_equity) / prior_equity
        except Exception as e:
            logger.error(f"Failed to read deploy ledger: {e}")
            return None

    def check_and_alert(self) -> Dict[str, Any]:
        """
        Main monitoring check. Call daily (or on each audit cycle).
        Returns a status dict. Human review is triggered but no autonomous action is taken.
        """
        band = self.load_crucible_band()
        daily_return = self.get_latest_daily_return()

        if band is None or daily_return is None:
            return {"status": "NO_DATA", "daily_return": daily_return, "band": band}

        in_band = band["lower"] <= daily_return <= band["upper"]

        if not in_band:
            self._consecutive_ood_days += 1
            logger.warning(
                f"[OUT-OF-BAND] Day {self._consecutive_ood_days}/{self.out_of_band_threshold}. "
                f"Daily return: {daily_return:.4f} | Band: [{band['lower']:.4f}, {band['upper']:.4f}]"
            )
            alert = {
                "timestamp": time.time(),
                "daily_return": daily_return,
                "band_lower": band["lower"],
                "band_upper": band["upper"],
                "consecutive_ood_days": self._consecutive_ood_days,
            }
            self._alert_log.append(alert)

            if self._consecutive_ood_days >= self.out_of_band_threshold:
                logger.critical(
                    f"[HUMAN REVIEW REQUIRED] Deployed agent has been out-of-band for "
                    f"{self._consecutive_ood_days} consecutive days. "
                    f"Mean return: {band['mean']:.4f} +/- {self.sigma_multiplier}*{band['std']:.4f}. "
                    f"Current: {daily_return:.4f}. MONITORING ONLY - no autonomous action."
                )
                return {
                    "status": "HUMAN_REVIEW_REQUIRED",
                    "consecutive_ood_days": self._consecutive_ood_days,
                    "daily_return": daily_return,
                    "band": band,
                    "alert_log": self._alert_log[-5:],
                }
        else:
            if self._consecutive_ood_days > 0:
                logger.info(f"[IN-BAND RESTORED] Return back in band after {self._consecutive_ood_days} days out.")
            self._consecutive_ood_days = 0

        return {
            "status": "IN_BAND" if in_band else "OUT_OF_BAND",
            "daily_return": daily_return,
            "band": band,
            "consecutive_ood_days": self._consecutive_ood_days,
        }

    def update_crucible_band(self, daily_returns: List[float]):
        """
        Called after each Crucible run to update the empirical band.
        Saves mean +/- std of daily returns to crucible_stats.json.
        """
        if not daily_returns:
            return
        n = len(daily_returns)
        mean = sum(daily_returns) / n
        variance = sum((r - mean) ** 2 for r in daily_returns) / max(n - 1, 1)
        std = math.sqrt(variance)
        os.makedirs(os.path.dirname(self.crucible_stats_path) or ".", exist_ok=True)
        with open(self.crucible_stats_path, "w") as f:
            json.dump({
                "mean_daily_return": mean,
                "std_daily_return": std,
                "n_samples": n,
                "updated_at": time.time(),
                "band_2sigma_lower": mean - 2 * std,
                "band_2sigma_upper": mean + 2 * std,
            }, f, indent=2)
        logger.info(f"Crucible band updated: mean={mean:.4f}, std={std:.4f}, n={n}")


if __name__ == "__main__":
    monitor = DeploymentMonitor(
        deploy_ledger_path="state/child_0/ledger.sqlite",
        crucible_stats_path="state/crucible_stats.json",
        out_of_band_threshold=3
    )
    # Simulate Crucible results to populate the band
    import random
    crucible_returns = [random.gauss(0.003, 0.012) for _ in range(30)]
    monitor.update_crucible_band(crucible_returns)
    
    status = monitor.check_and_alert()
    print("Monitoring status:", json.dumps(status, indent=2))
