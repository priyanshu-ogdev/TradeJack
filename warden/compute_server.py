"""
Warden Compute API Server: Fast-API/HTTP Endpoint for Child `dgx_compute_skill` Petitions.
Child agents petition `/petition_vram` before spawning PyTorch training jobs.
The Warden evaluates their SQLite ledger metrics and Grace Blackwell memory quotas, granting, throttling, or denying requests.
"""

import os
import sys
import time
import json
import logging
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
from typing import Dict, Any, Optional

# Import Warden core components
from warden.warden_core import WardenHypervisor, ContainerLedgerSummary
from warden.oom_watchdog import RecklessnessWatchdog
from warden.unified_memory_swap import BlackwellUnifiedAllocator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] (ComputeServer) %(message)s")
logger = logging.getLogger("ComputeServer")


class WardenAPIHandler(BaseHTTPRequestHandler):
    """
    HTTP Request Handler for Warden API service.
    """
    warden: WardenHypervisor = None
    watchdog: RecklessnessWatchdog = None
    unified_allocator: BlackwellUnifiedAllocator = None

    def _send_json_response(self, status_code: int, payload: Dict[str, Any]):
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(payload, indent=2).encode("utf-8"))

    def do_GET(self):
        parsed_path = urlparse(self.path)
        
        if parsed_path.path == "/system_status":
            # Return overall swarm status and burn rate
            active_children = [s for s in self.warden.summaries.values() if s.is_alive]
            total_equity = sum(s.equity for s in active_children)
            tier_counts = {1: 0, 2: 0, 3: 0}
            for s in active_children:
                tier_counts[s.tier] = tier_counts.get(s.tier, 0) + 1
                
            payload = {
                "timestamp": time.time(),
                "active_containers": len(active_children),
                "total_swarm_equity": total_equity,
                "tier_distribution": tier_counts,
                "base_tax_per_hr": self.warden.base_tax_per_hr,
                "dgx_spark_memory_status": {
                    "total_unified_memory_gb": 128.0,
                    "allocated_vram_gb": sum(20.0 if s.tier == 1 else (4.0 if s.tier == 2 else 0.0) for s in active_children)
                }
            }
            self._send_json_response(200, payload)
        elif parsed_path.path == "/health":
            self._send_json_response(200, {"status": "ONLINE", "service": "WardenComputeServer"})
        else:
            self._send_json_response(404, {"error": "Endpoint not found"})

    def do_POST(self):
        parsed_path = urlparse(self.path)
        content_length = int(self.headers.get("Content-Length", 0))
        body_str = self.rfile.read(content_length).decode("utf-8") if content_length > 0 else "{}"
        
        try:
            body = json.loads(body_str)
        except json.JSONDecodeError:
            self._send_json_response(400, {"error": "Invalid JSON body"})
            return
            
        if parsed_path.path == "/petition_vram":
            child_id = body.get("child_id")
            requested_gb = float(body.get("requested_vram_gb", 20.0))
            reason = body.get("reason", "Standard Model Training")
            
            if child_id is None:
                self._send_json_response(400, {"error": "Missing 'child_id' in petition"})
                return
                
            # Audit child ledger
            summary = self.warden.audit_child_ledger(int(child_id))
            if not summary or not summary.is_alive:
                self._send_json_response(403, {
                    "status": "DENIED",
                    "vram_limit_gb": 0.0,
                    "message": f"Child {child_id} is dead or insolvent. Training denied."
                })
                return
                
            # Evaluate tier
            tier = self.warden.assign_memory_tier(int(child_id), summary)
            
            if tier == 1 and requested_gb <= 20.0:
                self._send_json_response(200, {
                    "status": "GRANTED",
                    "vram_limit_gb": 20.0,
                    "tier": 1,
                    "expires_in_seconds": 3600,
                    "message": "High Alpha tier verified. 20GB MIG partition allocated."
                })
            elif tier == 2:
                self._send_json_response(200, {
                    "status": "THROTTLED",
                    "vram_limit_gb": 4.0,
                    "tier": 2,
                    "message": "Stagnant ledger. VRAM throttled to 4GB. You MUST use Dilated-CNN-Seq2seq or lightweight models."
                })
            else:
                self._send_json_response(200, {
                    "status": "DENIED",
                    "vram_limit_gb": 0.0,
                    "tier": 3,
                    "message": "Failing tier or OOM Watchdog lock active. Training denied. Inference-only micro-scalping permitted."
                })
        elif parsed_path.path == "/run_audit":
            tier_map = self.warden.run_audit_cycle()
            self._send_json_response(200, {"status": "SUCCESS", "tier_map": tier_map})
        else:
            self._send_json_response(404, {"error": "Endpoint not found"})

    def log_message(self, format, *args):
        logger.debug(format % args)


class WardenComputeServer:
    """
    Launches and manages the HTTP API daemon for Warden compute skill petitions.
    """

    def __init__(
        self,
        port: int = 8080,
        state_dir: str = "d:/TradeJack/state",
        swarm_size: int = 50
    ):
        self.port = port
        self.warden = WardenHypervisor(swarm_size=swarm_size, state_dir=state_dir)
        self.watchdog = RecklessnessWatchdog(state_dir=state_dir)
        self.unified_allocator = BlackwellUnifiedAllocator(vram_quota_gb=20.0)
        
        WardenAPIHandler.warden = self.warden
        WardenAPIHandler.watchdog = self.watchdog
        WardenAPIHandler.unified_allocator = self.unified_allocator
        self.server = HTTPServer(("0.0.0.0", self.port), WardenAPIHandler)

    def start(self):
        """Starts the API server and OOM watchdog."""
        logger.info(f"Starting Warden Compute Server on port {self.port}...")
        self.watchdog.start_monitoring()
        try:
            self.server.serve_forever()
        except KeyboardInterrupt:
            logger.info("Shutting down Warden Compute Server...")
        finally:
            self.watchdog.stop_monitoring()
            self.server.server_close()


if __name__ == "__main__":
    logger.info("Testing Warden Compute Server initialization...")
    server = WardenComputeServer(port=8085, swarm_size=5)
    # Run a single audit pass instead of blocking forever in test
    server.warden.init_child_ledger(0, initial_cash=25.0)
    summary = server.warden.audit_child_ledger(0)
    print("Initial ledger audit:", summary)
