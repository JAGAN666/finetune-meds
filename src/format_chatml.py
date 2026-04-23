"""Wrap raw {input, target} examples in Qwen2.5's chat template.

Single place where the system prompt is injected. If the schema or the
instructions change, `schema.SYSTEM_PROMPT` is the only thing to edit —
this module just plumbs it through.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from schema import SYSTEM_PROMPT  # noqa: E402


def to_messages(example: dict) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": example["input"]},
        {"role": "assistant", "content": example["target"]},
    ]


def to_chat_string(example: dict, tokenizer) -> str:
    """For TRL SFTTrainer with `dataset_text_field`."""
    return tokenizer.apply_chat_template(
        to_messages(example), tokenize=False, add_generation_prompt=False
    )


def format_for_training(example: dict, tokenizer) -> dict:
    return {"text": to_chat_string(example, tokenizer)}


def build_inference_prompt(input_text: str, tokenizer) -> str:
    """Build the prompt for inference (no assistant turn, add generation prompt)."""
    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": input_text},
    ]
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
