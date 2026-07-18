"""
DGX Compute Skill (`dgx_compute_skill` adaptation for `automaton`).
Exposes API methods for Child containers to petition the Warden for VRAM quotas and check system status.
"""

import os
import sys
import json
import logging
import requests
from typing import Dict, Any, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] (DGXComputeSkill) %(message)s")
logger = logging.getLogger("DGXComputeSkill")


class DGXComputeSkill:
    """
    Automaton skill allowing sovereign Child agents to query hardware capabilities,
    petition for Grace Blackwell MIG slices, and check competitive tier rankings.
    """

    def __init__(self, warden_host: str = "http://localhost:8080", child_id: int = 0):
        self.warden_host = warden_host.rstrip("/")
        self.child_id = child_id

    def get_system_status(self) -> Dict[str, Any]:
        """Queries current DGX memory pressure and burn rate."""
        try:
            resp = requests.get(f"{self.warden_host}/system_status", timeout=3.0)
            if resp.status_code == 200:
                return resp.json()
            return {"error": f"Status {resp.status_code}", "raw": resp.text}
        except Exception as e:
            logger.debug(f"Warden API not reachable ({e}). Returning local simulated status.")
            return {
                "status": "SIMULATED_LOCAL",
                "active_containers": 50,
                "total_swarm_equity": 500.0,
                "base_tax_per_hr": 1.0,
                "dgx_spark_memory_status": {"total_unified_memory_gb": 128.0, "allocated_vram_gb": 20.0}
            }

    def petition_for_vram(
        self,
        requested_vram_gb: float = 20.0,
        reason: str = "Breeding Gen-2 weights via neuro-evolution-novelty-search"
    ) -> Dict[str, Any]:
        """
        Petitions the Warden for VRAM quota before running PyTorch training jobs.
        """
        payload = {
            "child_id": self.child_id,
            "requested_vram_gb": requested_vram_gb,
            "reason": reason
        }
        try:
            resp = requests.post(f"{self.warden_host}/petition_vram", json=payload, timeout=5.0)
            if resp.status_code in [200, 403]:
                return resp.json()
            return {"status": "ERROR", "message": f"HTTP {resp.status_code}"}
        except Exception as e:
            logger.debug(f"Warden API petition failed ({e}). Returning simulated local grant based on requested amount.")
            # In local offline simulation without Warden running, default to granting requested amount or tier 2
            if requested_vram_gb > 10.0:
                return {
                    "status": "GRANTED_SIMULATED",
                    "vram_limit_gb": requested_vram_gb,
                    "tier": 1,
                    "message": "Local simulation mode: granted requested VRAM."
                }
            return {
                "status": "THROTTLED_SIMULATED",
                "vram_limit_gb": 4.0,
                "tier": 2,
                "message": "Local simulation mode: throttled to 4GB."
            }


if __name__ == "__main__":
    skill = DGXComputeSkill(child_id=1)
    print("Status test:", json.dumps(skill.get_system_status(), indent=2))
    print("Petition test:", json.dumps(skill.petition_for_vram(20.0), indent=2))
