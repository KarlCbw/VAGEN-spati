from vagen.env.base.base_env_config import BaseEnvConfig
from dataclasses import dataclass, fields
from typing import Optional, List, Tuple
import os

@dataclass
class EmbodiedVstarEnvConfig(BaseEnvConfig):
    env_name: str = "embodied_vstar"
    resolution: int = 1080  # Default resolution for the environment
    eval_set:str = "mixed"
    data_path = os.path.join(os.path.dirname(__file__), f"datasets/{eval_set}")
    
    render_mode: str = "vision" 
    prompt_format: str = "free_think"  # "free_think", "no_think", "grounding"
    max_actions_per_step: int = 1
    max_action_penalty: float = -1
    format_reward: float = 0.5
    effective_reward_weight: float = 1
    ineffective_penalty_weight: float = 0
    traj_success_reward: float = 1
    traj_fail_penalty: float = 0
    # "free_think", "no_think", "grounding"
    # configs for process reward (if needed in future)
    use_state_reward: bool = False
    grounding_reward_weight: float = 0.5
    worldmodeling_reward_weight: float = 0.5
    yaw_tolerance: int = 30
    pitch_tolerance: int = 20
    step_tolerance: int = 4
    def config_id(self) -> str:
        return f"EmbodiedVstarEnvConfig(eff_w={self.effective_reward_weight}, ineff_w={self.ineffective_penalty_weight}, succ_rw={self.traj_success_reward}, fail_rw={self.traj_fail_penalty})"

if __name__ == "__main__":
    config = EmbodiedVstarEnvConfig()
    print(config.config_id())


