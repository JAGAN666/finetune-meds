"""Grammar-constrained generation via `outlines`.

Layered on top of the fine-tuned model to show the production story:
fine-tuning gets you near-perfect content, constrained decoding gets
you a structural guarantee. Together → 100% valid JSON at inference.

`outlines.models.transformers` wraps a HF model; `outlines.generate.json`
returns a generator that only emits tokens consistent with the Pydantic
schema.

If outlines can't handle the quantized Unsloth model (this has happened
on specific bnb versions), the fallback is `lm-format-enforcer`, which
hooks into HF `generate()` directly via a logits processor.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).parent.parent))
from schema import MedicationList, SYSTEM_PROMPT  # noqa: E402


def generate_outlines(model_hf, tokenizer, inputs: List[str]) -> List[str]:
    import outlines
    om = outlines.models.Transformers(model_hf, tokenizer)
    gen = outlines.generate.json(om, MedicationList)
    outputs: List[str] = []
    for text in inputs:
        prompt = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": text},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        result = gen(prompt)
        # outlines returns the MedicationList instance directly; re-serialise
        outputs.append(result.model_dump_json())
    return outputs


def generate_lmfe(model_hf, tokenizer, inputs: List[str], max_new_tokens: int = 256) -> List[str]:
    """Fallback using lm-format-enforcer."""
    import torch
    from lmformatenforcer import JsonSchemaParser
    from lmformatenforcer.integrations.transformers import build_transformers_prefix_allowed_tokens_fn

    parser = JsonSchemaParser(MedicationList.model_json_schema())
    prefix_fn = build_transformers_prefix_allowed_tokens_fn(tokenizer, parser)

    model_hf.eval()
    device = next(model_hf.parameters()).device
    results: List[str] = []
    for text in inputs:
        prompt = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": text},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        enc = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model_hf.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                prefix_allowed_tokens_fn=prefix_fn,
                pad_token_id=tokenizer.eos_token_id,
            )
        new_tokens = out[0][enc["input_ids"].shape[1]:]
        results.append(tokenizer.decode(new_tokens, skip_special_tokens=True).strip())
    return results
