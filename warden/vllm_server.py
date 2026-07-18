"""
Shared Brain vLLM Server: High-Throughput Inference Endpoint with PagedAttention for 50 Sovereign Agents.
Manages KV-cache virtual memory blocks across a dedicated 32GB Grace Blackwell VRAM slice.
Enforces per-Child token rate limits tied to their competitive tier.
"""

import os
import sys
import time
import json
import logging
import subprocess
import requests
from typing import Dict, Any, Optional, List

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] (vLLMServer) %(message)s")
logger = logging.getLogger("vLLMServer")


class SharedBrainvLLM:
    """
    Manages the shared LLM endpoint (Llama-3.3-70B-Instruct or DeepSeek-R1-Distill) running on vLLM.
    Uses PagedAttention to prevent KV-cache OOM across 50 concurrent agent reasoning loops.
    """

    def __init__(
        self,
        model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
        gpu_memory_utilization: float = 0.25,  # ~32GB out of 128GB Blackwell
        port: int = 8000,
        host: str = "0.0.0.0"
    ):
        self.model_name = model_name
        self.gpu_memory_utilization = gpu_memory_utilization
        self.port = port
        self.host = host
        self.base_url = f"http://localhost:{self.port}/v1/chat/completions"
        self._server_process: Optional[subprocess.Popen] = None
        
        # Token limit quotas per Tier per minute
        self.tier_token_quotas = {
            1: {"max_tokens_per_req": 2048, "reqs_per_min": 60},
            2: {"max_tokens_per_req": 1024, "reqs_per_min": 30},
            3: {"max_tokens_per_req": 512,  "reqs_per_min": 15}
        }
        self.child_rate_tracker: Dict[int, Dict[str, float]] = {}

    def launch_server(self):
        """Launches vLLM OpenAI-compatible API server in a background subprocess."""
        if self._server_process and self._server_process.poll() is None:
            logger.info("vLLM server is already running.")
            return
            
        cmd = [
            sys.executable, "-m", "vllm.entrypoints.openai.api_server",
            "--model", self.model_name,
            "--gpu-memory-utilization", str(self.gpu_memory_utilization),
            "--port", str(self.port),
            "--host", self.host,
            "--max-num-batched-tokens", "8192",
            "--tensor-parallel-size", "1"
        ]
        
        logger.info(f"Launching vLLM Shared Brain with PagedAttention: {' '.join(cmd)}")
        try:
            # Launch in background; in production environment on Grace Blackwell this binds to CUDA
            self._server_process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            logger.info("vLLM server subprocess initialized.")
        except Exception as e:
            logger.error(f"Failed to launch vLLM server: {e}")

    def stop_server(self):
        """Terminates vLLM server subprocess."""
        if self._server_process:
            self._server_process.terminate()
            try:
                self._server_process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                self._server_process.kill()
            logger.info("vLLM server terminated.")

    def check_health(self) -> bool:
        """Checks if vLLM endpoint is responding."""
        try:
            resp = requests.get(f"http://localhost:{self.port}/health", timeout=2.0)
            return resp.status_code == 200
        except Exception:
            return False

    def check_rate_limit(self, child_id: int, tier: int) -> bool:
        """Enforces tier-based request frequency limits."""
        current_time = time.time()
        quota = self.tier_token_quotas.get(tier, self.tier_token_quotas[2])
        
        if child_id not in self.child_rate_tracker:
            self.child_rate_tracker[child_id] = {"window_start": current_time, "count": 0}
            
        tracker = self.child_rate_tracker[child_id]
        if current_time - tracker["window_start"] >= 60.0:
            tracker["window_start"] = current_time
            tracker["count"] = 0
            
        if tracker["count"] >= quota["reqs_per_min"]:
            logger.warning(f"Child {child_id} (Tier {tier}) exceeded rate limit of {quota['reqs_per_min']} reqs/min.")
            return False
            
        tracker["count"] += 1
        return True

    def query(
        self,
        child_id: int,
        tier: int,
        messages: List[Dict[str, str]],
        temperature: float = 0.3
    ) -> Optional[str]:
        """
        Sends an LLM reasoning prompt from a Child agent to the shared vLLM endpoint.
        Falls back to simulated/mock reasoning response if local server isn't live during bootstrap.
        """
        if not self.check_rate_limit(child_id, tier):
            return "ERROR: Rate limit exceeded for your resource tier."
            
        quota = self.tier_token_quotas.get(tier, self.tier_token_quotas[2])
        max_tokens = quota["max_tokens_per_req"]
        
        payload = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        
        try:
            resp = requests.post(self.base_url, json=payload, timeout=30.0)
            if resp.status_code == 200:
                data = resp.json()
                return data["choices"][0]["message"]["content"]
            else:
                logger.error(f"vLLM returned status {resp.status_code}: {resp.text}")
                return self._fallback_reasoning(child_id, tier, messages)
        except Exception as e:
            logger.debug(f"vLLM server not reachable ({e}). Using mock/fallback reasoning engine.")
            return self._fallback_reasoning(child_id, tier, messages)

    def _fallback_reasoning(self, child_id: int, tier: int, messages: List[Dict[str, str]]) -> str:
        """
        Fallback reasoning logic when running in local development without live 70B GPU weights loaded.
        Simulates structured autonomous decision output for Think -> Act -> Observe loop.
        """
        last_msg = messages[-1]["content"] if messages else ""
        if "VRAM is throttled" in last_msg or tier == 2:
            return json.dumps({
                "thought": "I am in Tier 2 (Stagnant, 4GB VRAM). My heavy Transformer models will throw OOM. I will adapt using self-mod/ to rewrite inference to Dilated-CNN-Seq2seq and execute medium-frequency mean reversion.",
                "action": "SELF_MOD_SWAP_CNN",
                "model": "Dilated-CNN-Seq2seq"
            })
        elif tier == 1:
            return json.dumps({
                "thought": "I am in Tier 1 (High Alpha, 20GB VRAM). I will run neuro-evolution-novelty-search to breed Gen-2 weights and broadcast alpha over social/ relay for USDC.",
                "action": "TRAIN_NEURO_EVOLUTION",
                "model": "Attention-is-all-you-Need"
            })
        else:
            return json.dumps({
                "thought": "I am in Tier 3 (Failing/Inference Only, 0GB Train VRAM). I cannot evolve. I must execute high-variance Deep-Q-learning micro-scalping to recover funds.",
                "action": "MICRO_SCALP_Q_LEARNING",
                "model": "Deep-Q-learning"
            })


if __name__ == "__main__":
    server = SharedBrainvLLM()
    res = server.query(1, 1, [{"role": "user", "content": "Analyze market and decide strategy."}])
    print("Test reasoning response:", res)
