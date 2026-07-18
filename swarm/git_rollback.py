"""
Git Financial Rollback Engine (`GitFinancialRollback`).
Automates High-Water Mark (HWM) local Git tagging (`git tag -a vX.Y`) when equity hits new peaks,
and instantly executes `git checkout vX.Y` upon suffering >15% drawdown to revert neural weights and
state parameters back to their last proven profitable state.
"""

import os
import sys
import time
import logging
import subprocess
from typing import Dict, Any, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] (GitRollback) %(message)s")
logger = logging.getLogger("GitRollback")


class GitFinancialRollback:
    """
    Sovereign survival checkpoint and rollback mechanism.
    """

    def __init__(
        self,
        child_id: int = 0,
        repo_dir: str = "d:/TradeJack",
        drawdown_rollback_threshold: float = 0.15,
        tag_step_dollar: float = 1.0
    ):
        self.child_id = child_id
        self.repo_dir = os.path.abspath(repo_dir)
        self.drawdown_threshold = drawdown_rollback_threshold
        self.tag_step_dollar = tag_step_dollar
        
        self.hwm_equity = 10.0
        self.last_tagged_equity = 10.0
        self.tag_version = 0
        self.latest_tag_name: Optional[str] = None

    def check_and_checkpoint(self, current_equity: float, weights_file: Optional[str] = None) -> Optional[str]:
        """
        Evaluates equity against high-water mark. If a significant gain has occurred (> $1.00 above last tag),
        tags the repository and saves the state.
        """
        if current_equity > self.hwm_equity:
            self.hwm_equity = current_equity
            
        if current_equity - self.last_tagged_equity >= self.tag_step_dollar:
            self.tag_version += 1
            tag_name = f"v_child{self.child_id}_{self.tag_version}"
            self.latest_tag_name = tag_name
            self.last_tagged_equity = current_equity
            
            # Save weights checkpoint if provided
            state_dir = os.path.join(self.repo_dir, "state", f"child_{self.child_id}")
            os.makedirs(state_dir, exist_ok=True)
            
            if weights_file and os.path.exists(weights_file):
                import shutil
                dest = os.path.join(state_dir, f"weights_{tag_name}.pt")
                shutil.copy2(weights_file, dest)
                logger.debug(f"Copied proven weights to {dest}")
                
            # Execute Git tag (if in a live Git repository or simulation log)
            try:
                subprocess.run(
                    ["git", "tag", "-f", "-a", tag_name, "-m", f"HWM Equity ${current_equity:.2f}"],
                    cwd=self.repo_dir, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False
                )
                logger.info(f"[HWM CHECKPOINT] Tagged repository state with '{tag_name}' at Equity ${current_equity:.2f}.")
            except Exception as e:
                logger.debug(f"Git CLI tag failed or not repo ({e}). Recorded local tag checkpoint.")
                
            return tag_name
        return None

    def execute_rollback_if_breached(
        self,
        current_equity: float,
        current_drawdown: float,
        active_weights_path: Optional[str] = None
    ) -> bool:
        """
        If `current_drawdown >= 0.15` (>15% loss from peak), executes instant git rollback to last HWM tag.
        Returns True if rollback occurred.
        """
        if current_drawdown >= self.drawdown_threshold and self.latest_tag_name is not None:
            logger.warning(
                f"[DRAWDOWN BREACH {current_drawdown*100:.1f}% >= {self.drawdown_threshold*100:.1f}%] "
                f"Executing instant financial rollback to HWM tag '{self.latest_tag_name}' (${self.last_tagged_equity:.2f})."
            )
            
            state_dir = os.path.join(self.repo_dir, "state", f"child_{self.child_id}")
            backup_weights = os.path.join(state_dir, f"weights_{self.latest_tag_name}.pt")
            
            if active_weights_path and os.path.exists(backup_weights):
                import shutil
                shutil.copy2(backup_weights, active_weights_path)
                logger.info(f"Successfully restored weights file {active_weights_path} from HWM backup {backup_weights}.")
                
            # Attempt git checkout of state folder from tag
            try:
                subprocess.run(
                    ["git", "checkout", self.latest_tag_name, "--", f"state/child_{self.child_id}"],
                    cwd=self.repo_dir, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False
                )
            except Exception:
                pass
                
            # Reset HWM tracking after rollback
            self.hwm_equity = self.last_tagged_equity
            return True
        return False


if __name__ == "__main__":
    logger.info("Testing GitFinancialRollback standalone execution...")
    rollback = GitFinancialRollback(child_id=1, tag_step_dollar=0.5)
    
    # Simulate equity gain triggering checkpoint
    rollback.check_and_checkpoint(10.60)
    print("Latest tag after $10.60:", rollback.latest_tag_name)
    
    # Simulate drawdown triggering rollback
    did_rollback = rollback.execute_rollback_if_breached(current_equity=8.80, current_drawdown=0.17)
    print("Did rollback trigger on 17% drawdown:", did_rollback)
