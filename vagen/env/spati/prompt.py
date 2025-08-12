# -*- coding: utf-8 -*-
"""
Prompt utilities for two-view video MCQ (Spati) environment.

This file defines:
- FORMAT_CONFIGS: response styles and examples
- system_prompt(): global system instruction + optional example
- init_observation_template() / action_template(): per-turn rendering
- format_prompt_generator() and format_prompt: per-format add-on prompt
"""

# -------------------------------
# 1) Response format configurations
# -------------------------------
FORMAT_CONFIGS = {
    # Default: no chain-of-thought, return only the final option.
    "mvqa_mcq": {
        "description": (
            "This is a two-view video multiple-choice question (MCQ). "
            "Return only one choice using the <answer> tag."
        ),
        "format": "<answer>A|B|C|D</answer>",
        "example": (
            # Minimal example with two placeholders and four options.
            "[View 1] <view1>\n"
            "[View 2] <view2>\n"
            "Q: Which option is correct based on the motion cues?\n"
            "A) 60°\nB) 120°\nC) 30°\nD) 180°\n"
            "<answer>C</answer>"
        ),
    },

    # Explicit no-think style (identical to mvqa_mcq, separate key for clarity).
    "mvqa_mcq_no_think": {
        "description": (
            "Two-view video MCQ. Provide only the final answer using the <answer> tag. "
            "Do not include any other text."
        ),
        "format": "<answer>A|B|C|D</answer>",
        "example": (
            "[View 1] <view1>\n"
            "[View 2] <view2>\n"
            "Q: Select the relative angle between cameras.\n"
            "A) 60°\nB) 120°\nC) 30°\nD) 180°\n"
            "<answer>B</answer>"
        ),
    },

    # Optional free-think style (if you want to allow a short rationale).
    "mvqa_mcq_free_think": {
        "description": (
            "Two-view video MCQ. First write a brief reasoning inside <think>, "
            "then return only one choice in <answer>."
        ),
        "format": "<think>...</think>\n<answer>A|B|C|D</answer>",
        "example": (
            "[View 1] <view1>\n"
            "[View 2] <view2>\n"
            "Q: Which option is correct based on motion parallax and trajectory alignment?\n"
            "A) 60°\nB) 120°\nC) 30°\nD) 180°\n"
            "<think>The flow directions imply approximately opposite headings; thus 180°.</think>\n"
            "<answer>D</answer>"
        ),
    },
}


# -------------------------------
# 2) System prompt
# -------------------------------
def system_prompt(**kwargs):
    """
    Compose a global system instruction for the two-view MCQ task.

    Args:
        format (str): One of FORMAT_CONFIGS keys. Defaults to "mvqa_mcq".
                      Controls whether examples show <think> or not.

    Returns:
        str: The system prompt shown to the model before turns.
    """
    fmt_key = kwargs.get("format", "mvqa_mcq")
    if fmt_key not in FORMAT_CONFIGS:
        fmt_key = "mvqa_mcq"

    # Base rules tailored for two-view MCQ with strict output constraint.
    base = f"""You are answering a two-view video multiple-choice question (MCQ).
You will receive two short videos captured from different viewpoints: <view1> and <view2>.
Carefully compare motions, occlusions, and parallax patterns across both views to answer the question.

Rules:
1) You must return exactly one choice using the <answer> tag. The only valid outputs are:
   <answer>A</answer> or <answer>B</answer> or <answer>C</answer> or <answer>D</answer>.
2) Do NOT include extra text outside the specified format. No tool calls or code.
3) If uncertain, select the most likely option.
4) This is a single-step task; once you answer, the episode ends.
"""

    # Provide an inline example consistent with the chosen format.
    example = f"Example:\n{FORMAT_CONFIGS[fmt_key]['example']}"
    return base + "\n" + example


# -------------------------------
# 3) Turn templates
# -------------------------------
def init_observation_template(**kwargs):
    """
    Render the initial observation with two video placeholders and the MCQ.

    Expected kwargs:
        observation_view1 (str): Placeholder for view 1 (e.g., "<view1>")
        observation_view2 (str): Placeholder for view 2 (e.g., "<view2>")
        question (str): Question text (optional; can already contain placeholders)
        options (List[str]): Four options in order [A_text, B_text, C_text, D_text]
        instruction (str): Additional instruction to emphasize output rules

    Returns:
        str: The initial observation string.
    """
    v1 = kwargs.get("observation_view1", "<view1>")
    v2 = kwargs.get("observation_view2", "<view2>")
    question = kwargs.get("question", "No question provided.")
    options = kwargs.get("options", ["(A)", "(B)", "(C)", "(D)"])
    instruction = kwargs.get(
        "instruction",
        "Return exactly one option using <answer> tag, e.g., <answer>B</answer>."
    )

    # Normalize options to A-D lines.
    opt_lines = []
    labels = ["A", "B", "C", "D"]
    for i, txt in enumerate(options[:4]):
        opt_lines.append(f"{labels[i]}) {txt}")

    return (
        f"[View 1] {v1}\n"
        f"[View 2] {v2}\n"
        f"Question:\n{question}\n"
        f"Options:\n" + "\n".join(opt_lines) + "\n"
        f"{instruction}"
    )


def action_template(**kwargs):
    """
    Render a follow-up turn. Usually not needed for single-step MCQ, but kept
    for compatibility with the framework (e.g., feedback on invalid format).

    Expected kwargs:
        observation_view1 (str)
        observation_view2 (str)
        question (str)
        options (List[str])
        env_feedback (str)
        done (bool)

    Returns:
        str: The action turn string.
    """
    v1 = kwargs.get("observation_view1", "<view1>")
    v2 = kwargs.get("observation_view2", "<view2>")
    question = kwargs.get("question", "No question provided.")
    options = kwargs.get("options", ["(A)", "(B)", "(C)", "(D)"])
    env_feedback = kwargs.get("env_feedback", "")
    done = kwargs.get("done", False)

    opt_lines = []
    labels = ["A", "B", "C", "D"]
    for i, txt in enumerate(options[:4]):
        opt_lines.append(f"{labels[i]}) {txt}")

    if done:
        return (
            f"Environment feedback: {env_feedback}\n"
            f"Task is done."
        )

    return (
        f"Environment feedback: {env_feedback}\n"
        f"[View 1] {v1}\n"
        f"[View 2] {v2}\n"
        f"Question:\n{question}\n"
        f"Options:\n" + "\n".join(opt_lines) + "\n"
        f"Return exactly one option in <answer> tag."
    )


# -------------------------------
# 4) Format prompt generator
# -------------------------------
def format_prompt_generator(format_key):
    """
    Create a per-format prompt function used as an add-on under each turn.

    The generated function accepts:
        - max_actions_per_step (int): For display only. Keep as 1 for MCQ.
        - add_example (bool): If True, append a small in-format example.

    Returns:
        Callable[..., str]
    """
    def prompt_function(**kwargs):
        max_actions_per_step = kwargs.get("max_actions_per_step", 1)
        add_example = kwargs.get("add_example", True)

        if format_key not in FORMAT_CONFIGS:
            raise ValueError(f"Unknown format key: {format_key}")
        cfg = FORMAT_CONFIGS[format_key]

        head = (
            f"You can take {max_actions_per_step} action(s) at a time (MCQ is single-step).\n"
            f"{cfg['description']}\n"
            f"Your response MUST follow:\n{cfg['format']}"
        )
        if add_example:
            return head + "\n" + "e.g. " + cfg["example"]
        return head

    return prompt_function


# Expose a dict, same pattern as your original file.
format_prompt = {
    key: format_prompt_generator(key)
    for key in FORMAT_CONFIGS
}


# -------------------------------
# 5) Local test
# -------------------------------
if __name__ == "__main__":
    for key, func in format_prompt.items():
        print(f"=== system_prompt({key}) ===")
        print(system_prompt(format=key))
        print("\n--- format_prompt ---")
        print(func(max_actions_per_step=1, add_example=True))
        print("\n" + "=" * 60 + "\n")
