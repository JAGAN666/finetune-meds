"""Vanilla generation wrapper — used for both the base-model baselines
and the fine-tuned model. The only difference between the three eval
runs (zero-shot / 3-shot / fine-tuned) is the model passed in and the
optional few-shot prefix.

Returns the raw assistant text — eval.py's `_safe_parse` handles JSON
extraction and cleanup.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from schema import SYSTEM_PROMPT  # noqa: E402


FEW_SHOT_EXAMPLES = [
    {
        "input": "Start metoprolol tartrate 25 mg PO BID for 2 weeks.",
        "target": '{"medications": [{"drug": "metoprolol tartrate", "dose": "25 mg", "route": "PO", "frequency": "BID", "duration": "2 weeks"}]}',
    },
    {
        "input": "Continue lisinopril 10 mg daily.",
        "target": '{"medications": [{"drug": "lisinopril", "dose": "10 mg", "route": null, "frequency": "daily", "duration": null}]}',
    },
    {
        "input": "Patient denies taking any OTC supplements.",
        "target": '{"medications": []}',
    },
]


@dataclass
class GenOptions:
    max_new_tokens: int = 256
    temperature: float = 0.0  # deterministic extraction
    do_sample: bool = False


def build_messages(input_text: str, n_shots: int = 0) -> list[dict]:
    msgs: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    for ex in FEW_SHOT_EXAMPLES[:n_shots]:
        msgs.append({"role": "user", "content": ex["input"]})
        msgs.append({"role": "assistant", "content": ex["target"]})
    msgs.append({"role": "user", "content": input_text})
    return msgs


def generate(
    model,
    tokenizer,
    inputs: List[str],
    n_shots: int = 0,
    opts: Optional[GenOptions] = None,
) -> List[str]:
    opts = opts or GenOptions()
    results: List[str] = []
    model.eval()
    device = next(model.parameters()).device
    for text in inputs:
        msgs = build_messages(text, n_shots=n_shots)
        prompt = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )
        enc = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=opts.max_new_tokens,
                do_sample=opts.do_sample,
                temperature=opts.temperature if opts.do_sample else 1.0,
                pad_token_id=tokenizer.eos_token_id,
            )
        new_tokens = out[0][enc["input_ids"].shape[1]:]
        results.append(tokenizer.decode(new_tokens, skip_special_tokens=True).strip())
    return results
