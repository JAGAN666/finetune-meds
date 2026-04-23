"""Synthetic medication-extraction example generator.

JSON-first: the LLM proposes the target JSON, then writes a natural
sentence that expresses it. This keeps labels correct by construction —
we don't ask the model to label its own free-form output, which would
let labelling errors into the training set.

Substitute-and-verify: every generated pair is validated by searching
for each non-null field's surface string inside the sentence. Pairs
that fail the invariant are dropped.

Targets the specific weak-spots MACCROBAT + BioLeaflets under-cover:
multi-drug sentences, no-med negatives, abbreviation-heavy prose,
ambiguous duration language.

Uses Claude Sonnet 4.6 with prompt caching so the schema + few-shot
block costs once, not per call. Typical budget: ~$5 for 1.5K examples.
"""

from __future__ import annotations

import json
import os
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

try:
    from anthropic import Anthropic
except ImportError:  # allow import without SDK for unit-testing the validator
    Anthropic = None  # type: ignore

MODEL = "claude-sonnet-4-6"

STYLES = ["prescription_order", "discharge_summary", "nursing_note", "patient_facing"]
DIFFICULTIES = [
    "single_med",
    "multi_med_3_to_5",
    "abbreviation_heavy",
    "ambiguous_duration",
    "no_medication",
]

SYSTEM = """You generate realistic clinical language snippets paired with strict JSON medication extractions.

Target schema:
{"medications": [{"drug": str, "dose": str|null, "route": str|null, "frequency": str|null, "duration": str|null}]}

Generation protocol (IMPORTANT):
1. First silently decide the medications list as JSON.
2. Then write one short clinical sentence or brief paragraph that expresses EXACTLY those medications and attributes — no extras, no omissions.
3. Every non-null field value in the JSON must appear as a substring inside the sentence, preserving the exact surface form.
4. Return a JSONL block. Each line is {"input": <sentence>, "target": <medications-object>}.
5. Never include markdown fences. Never wrap the output in quotes."""

FEW_SHOT = """Examples:
{"input": "Start metoprolol tartrate 25 mg PO BID for 2 weeks, then reassess.", "target": {"medications": [{"drug": "metoprolol tartrate", "dose": "25 mg", "route": "PO", "frequency": "BID", "duration": "2 weeks"}]}}
{"input": "Continue lisinopril 10 mg daily and add atorvastatin 40 mg at bedtime.", "target": {"medications": [{"drug": "lisinopril", "dose": "10 mg", "route": null, "frequency": "daily", "duration": null}, {"drug": "atorvastatin", "dose": "40 mg", "route": null, "frequency": "at bedtime", "duration": null}]}}
{"input": "Patient denies taking any OTC supplements or herbal remedies.", "target": {"medications": []}}"""


@dataclass
class GenConfig:
    n_total: int
    batch_size: int = 15
    style_weights: Optional[dict] = None
    difficulty_weights: Optional[dict] = None

    def pick_style(self) -> str:
        w = self.style_weights or {s: 1 for s in STYLES}
        return random.choices(list(w.keys()), weights=list(w.values()))[0]

    def pick_difficulty(self) -> str:
        # bias toward the underrepresented shapes
        default = {
            "single_med": 2,
            "multi_med_3_to_5": 3,
            "abbreviation_heavy": 3,
            "ambiguous_duration": 2,
            "no_medication": 1,
        }
        w = self.difficulty_weights or default
        return random.choices(list(w.keys()), weights=list(w.values()))[0]


def generate(n: int, out_path: Optional[Path] = None, seed: int = 7) -> List[dict]:
    """Generate ~n validated synthetic examples. Actual yield may be slightly
    lower because invariant-violating pairs are dropped."""
    random.seed(seed)
    if Anthropic is None:
        raise RuntimeError("anthropic SDK not installed; add to requirements.txt")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY not set")

    client = Anthropic()
    cfg = GenConfig(n_total=n)
    out: List[dict] = []
    seen_inputs: set[str] = set()
    while len(out) < n:
        batch = _call_once(client, cfg)
        for ex in batch:
            if ex["input"] in seen_inputs:
                continue
            if not _validate(ex):
                continue
            seen_inputs.add(ex["input"])
            out.append({
                "input": ex["input"],
                "target": json.dumps(ex["target"], ensure_ascii=False),
                "source": "synthetic",
                "doc_id": f"syn_{len(out):05d}",
            })
            if len(out) >= n:
                break
        print(f"  [synth] {len(out)}/{n}", file=sys.stderr)

    if out_path is not None:
        import pandas as pd
        pd.DataFrame(out).to_parquet(out_path, index=False)
    return out


def _call_once(client, cfg: GenConfig) -> List[dict]:
    style = cfg.pick_style()
    difficulty = cfg.pick_difficulty()
    user_msg = (
        f"Generate {cfg.batch_size} examples.\n"
        f"Style: {style}\n"
        f"Difficulty: {difficulty}\n"
        f"Return JSONL — one JSON object per line."
    )

    # Prompt caching: system + few-shot are marked ephemeral. They stay in
    # the cache across calls within a 5-minute window, so 100 calls cost
    # the system/few-shot tokens once.
    resp = client.messages.create(
        model=MODEL,
        max_tokens=2048,
        system=[
            {"type": "text", "text": SYSTEM, "cache_control": {"type": "ephemeral"}},
            {"type": "text", "text": FEW_SHOT, "cache_control": {"type": "ephemeral"}},
        ],
        messages=[{"role": "user", "content": user_msg}],
    )
    raw = resp.content[0].text
    return list(_parse_jsonl(raw))


def _parse_jsonl(raw: str):
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "input" in obj and "target" in obj:
            yield obj


def _validate(ex: dict) -> bool:
    """Substitute-and-verify: every non-null field must appear in the sentence."""
    sentence = ex["input"]
    target = ex["target"]
    meds = target.get("medications")
    if not isinstance(meds, list):
        return False
    for m in meds:
        if not isinstance(m, dict):
            return False
        if "drug" not in m or not m["drug"]:
            return False
        for field in ("drug", "dose", "route", "frequency", "duration"):
            v = m.get(field)
            if v is None:
                continue
            if v not in sentence:
                return False
    return True


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=1500)
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "synthetic.parquet")
    args = parser.parse_args()
    generate(args.n, out_path=args.out)
