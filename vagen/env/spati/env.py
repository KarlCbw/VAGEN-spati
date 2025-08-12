from vagen.env.base.base_env import BaseEnv
from pathlib import Path
from PIL import Image
import cv2
import json
from vagen.env.utils.parse_utils import PARSE_FUNC_MAP
from vagen.env.spati.prompt import (
    format_prompt, system_prompt, action_template, init_observation_template
)
from vagen.env.spati.env_config import SpatiEnvConfig

class SpatiEnv(BaseEnv):
    def __init__(self, config: SpatiEnvConfig):
        super().__init__()
        self.config = config
        self.dataset = self._load_dataset()
        self.format_prompt_func = format_prompt[self.config.prompt_format]
        self.parse_func = PARSE_FUNC_MAP[self.config.prompt_format]

        self._idx = 0
        self.total_reward = 0.0
        self.reward = 0.0
        self.done = False
        self.info = {}
        self.sample = None

        # NEW: round index
        self.round_idx = 0
        self.answered = False

    def _load_dataset(self):
        """Load QA samples from a jsonl file."""
        dataset = []
        p = Path(self.config.data_path)
        assert p.exists(), f"Dataset file {p} not found."
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                it = json.loads(line)
                # Required fields: video_1, video_2, question, options, answer
                assert all(k in it for k in ["video_1", "video_2", "question", "options", "answer"])
                dataset.append(it)
        return dataset

    def _load_video_frames(self, video_path):
        """Load all frames from a short video into a list of PIL Images."""
        cap = cv2.VideoCapture(str(video_path))
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(img))
        cap.release()
        return frames

    def _render(self, init_obs=True):
        """Render current observation text and two-view frames."""
        pf = self.format_prompt_func(add_example=False)
        v1, v2 = self.config.image_placeholder_v1, self.config.image_placeholder_v2

        frames_v1 = self._load_video_frames(self.sample["video_1"])
        frames_v2 = self._load_video_frames(self.sample["video_2"])
        multi = {v1: frames_v1, v2: frames_v2}

        q = self.sample["question"]
        opts = self.sample["options"]

        # Provide allowed action hint for the template
        valid_action = ["deliberate()", "submit(A|B|C|D)"]

        if init_obs:
            obs_str = init_observation_template(
                observation_view1=v1, observation_view2=v2,
                question=q, options=opts,
                instruction=(
                    "You may think for several rounds, then answer once. "
                    "Thinking rounds: return only <think>...</think>. "
                    "Final round: return <answer>A|B|C|D</answer> (no extra text)."
                )
            ) + "\n" + pf
        else:
            obs_str = action_template(
                observation_view1=v1, observation_view2=v2,
                question=q, options=opts,
                env_feedback=self.info.get("env_feedback", ""),
                done=self.done
            ) + "\n" + pf

        return {"obs_str": obs_str, "multi_modal_data": multi}

    def reset(self, seed=None):
        """Reset the episode and show initial two-view observation."""
        self._idx = (seed or 0) % len(self.dataset)
        self.sample = self.dataset[self._idx]
        self.reward = 0.0
        self.total_reward = 0.0
        self.done = False
        self.info = {"env_step": 0, "is_format_rewarded": False}
        self.round_idx = 0
        self.answered = False
        return self._render(init_obs=True), {}


    def step(self, llm_raw_response: str):
        """
        Multi-round logic:
        - If <answer> present -> submit(X) and finish.
        - Else -> deliberate() and continue until max_rounds.
        """
        rst = self.parse_func(
            response=llm_raw_response,
            special_token_list=self.config.__dict__.get("special_token_list", None),
            max_actions=self.config.max_actions_per_step,
            action_sep="|"
        )
        self.reward = 0.0
        done = False
        info = {}
        info.update(rst)

        is_valid = len(rst.get("actions", [])) == 1 and rst.get("format_correct", True)

        if is_valid:
            action = rst["actions"][0]
            try:
                if action.startswith("submit"):
                    # Final answer path
                    choice = action.replace("submit", "").strip("()").strip().upper()
                    idx_to_letter = ["A", "B", "C", "D"]
                    gt_letter = idx_to_letter[int(self.sample["answer"])]
                    if choice == gt_letter:
                        self.reward += self.config.traj_success_reward
                    else:
                        self.reward += self.config.traj_fail_penalty
                    done = True
                    self.answered = True
                    info["env_feedback"] = "Answer received."
                elif action.startswith("deliberate"):
                    # Thinking path (no answer yet)
                    # Optional step penalty after certain rounds
                    if self.round_idx >= self.config.delay_penalty_after:
                        self.reward -= self.config.per_step_penalty
                    info["env_feedback"] = "Deliberation noted. You may continue or submit your final answer."
                    done = False
                else:
                    is_valid = False
            except Exception:
                is_valid = False

        # If invalid format, encourage correct format on next round
        if not is_valid:
            info["env_feedback"] = "Invalid format. Use <think>...</think> for deliberation or <answer>X</answer> to submit."
            # Optionally penalize invalid formatting
            # self.reward -= 0.0

        # Bump round index and check max_rounds
        self.round_idx += 1
        if not done and self.round_idx >= self.config.max_rounds:
            # force finish if still no submission
            done = True
            if not self.answered:
                self.reward += self.config.traj_fail_penalty
                info["env_feedback"] = "Max rounds reached without an answer. Episode terminated."

        info["is_format_rewarded"] = bool(is_valid)
        self.info = info
        self.done = done
        self.total_reward += self.reward
        self.info["env_step"] = self.info.get("env_step", 0) + 1
        self.info["done"] = done

        return self._render(init_obs=False), self.reward, done, self.info

    def compute_reward(self) -> float:
        """Return the accumulated trajectory reward."""
        return float(self.total_reward)

    def system_prompt(self) -> str:
        """Return the system prompt for the model."""
        return system_prompt(format=self.config.prompt_format) + "\n" + \
               self.format_prompt_func(add_example=True)

    def close(self):
        """Clean up resources."""
        pass
