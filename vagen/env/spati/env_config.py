# -*- coding: utf-8 -*-
"""
Config for the two-view video MCQ environment: SpatiEnv.

- Inherit from BaseEnvConfig to stay consistent with VAGEN's config system.
- Default behavior: load ALL frames from short videos (no subsampling).
- Reward: single-step MCQ (+1 for correct, -1 for wrong by default).
- Prompt format: "mvqa_mcq" (see your spati/prompt.py).
"""

from vagen.env.base.base_env_config import BaseEnvConfig
from dataclasses import dataclass
from typing import Optional, List
import os

@dataclass
class SpatiEnvConfig(BaseEnvConfig):
    # --- Identity ---
    env_name: str = "spati"

    # --- Data source ---
    # Path to a jsonl file; each line contains:
    # {"video_1": "...mp4", "video_2": "...mp4",
    #  "question": "...", "options": ["...", "...", "...", "..."], "answer": 0-based-index}
    #
    # NOTE: Change this to your actual qa.jsonl path (or override via CLI).
    data_path: str = os.path.join("")

    # --- Rendering / modality flags (kept for consistency with VAGEN) ---
    render_mode: str = "vision"  # Vision-only; no text-only mode needed here.

    # --- Prompting ---
    # Must match a key in your spati/prompt.py FORMAT_CONFIGS (e.g., "mvqa_mcq")
    prompt_format: str = "mvqa_mcq"

    # Placeholders shown in the prompt and used as keys in multi_modal_data
    image_placeholder_v1: str = "<view1>"
    image_placeholder_v2: str = "<view2>"

    # --- Step constraints ---
    # MCQ is single-step: the model returns exactly one <answer> tag and the episode ends.
    max_actions_per_step: int = 1

    # --- Reward shaping ---
    # Single-step task: correct → +succ, wrong → +fail (usually negative).
    traj_success_reward: float = 1.0
    traj_fail_penalty: float = -1.0

    # Optional "format reward" if you use env_state_reward_wrapper to reward correct format.
    format_reward: float = 0.5

    # --- Parsing / tokenizer hints (optional) ---
    # If you have special tokens (e.g., "<answer>"), you can pass them here.
    special_token_list: Optional[List[str]] = None

    # --- Video loading policy ---
    # For short clips we load all frames. If you ever want subsampling, set load_all_frames=False
    # and specify frames_per_video > 0; your env can branch on these.
    load_all_frames: bool = True
    frames_per_video: int = -1  # Ignored when load_all_frames=True

    # --- Misc / debugging ---
    verbose: bool = False

    def config_id(self) -> str:
        """String ID summarizing the critical knobs; useful for logs/checkpoints."""
        return (
            f"SpatiEnvConfig("
            f"succ={self.traj_success_reward}, "
            f"fail={self.traj_fail_penalty}, "
            f"fmt={self.format_reward}, "
            f"prompt='{self.prompt_format}', "
            f"load_all={self.load_all_frames}"
            f")"
        )

if __name__ == "__main__":
    cfg = SpatiEnvConfig()
    print(cfg.config_id())
    print("data_path:", cfg.data_path)
