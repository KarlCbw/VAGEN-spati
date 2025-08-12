# -*- coding: utf-8 -*-
"""
Config for the two-view video MCQ environment: SpatiEnv.

- Inherit from BaseEnvConfig to stay consistent with VAGEN's config system.
- Default behavior: load ALL frames from short videos (no subsampling).
- Reward: single-step MCQ (+1 for correct, -1 for wrong by default).
- Prompt format: "mvqa_mcq" (see your spati/prompt.py).
"""

from dataclasses import dataclass
from vagen.env.base.base_env_config import BaseEnvConfig
from typing import Optional, List
import os

@dataclass
class SpatiEnvConfig(BaseEnvConfig):
    env_name: str = "spati"
    data_path: str = os.path.join(os.path.dirname(__file__), "datasets/mvqa/qa.jsonl")
    render_mode: str = "vision"

    # NEW: use a multi-round prompt
    prompt_format: str = "mvqa_mcq_multiround"

    image_placeholder_v1: str = "<view1>"
    image_placeholder_v2: str = "<view2>"
    max_actions_per_step: int = 1

    # Rewards
    traj_success_reward: float = 1.0
    traj_fail_penalty: float = -1.0
    format_reward: float = 0.5

    # NEW: multi-round control
    max_rounds: int = 4                # allow up to N rounds before final submit
    per_step_penalty: float = 0.0      # penalty per deliberate round
    delay_penalty_after: int = 1       # start penalizing after this round index (0-based)

    # Video loading
    load_all_frames: bool = True
    frames_per_video: int = -1

    special_token_list: Optional[List[str]] = None
    verbose: bool = False

    def config_id(self) -> str:
        return (
            f"SpatiEnvConfig(succ={self.traj_success_reward}, "
            f"fail={self.traj_fail_penalty}, fmt={self.format_reward}, "
            f"prompt='{self.prompt_format}', max_rounds={self.max_rounds})"
        )

if __name__ == "__main__":
    cfg = SpatiEnvConfig()
    print(cfg.config_id())
    print("data_path:", cfg.data_path)
