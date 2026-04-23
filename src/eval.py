"""Evaluation metrics for medication extraction.

All metrics are mechanical — no subjective judging. Each run of the eval
harness emits the same five numbers so the base / few-shot / fine-tuned
comparison is apples-to-apples.

Metrics:
  - JSON parse rate:     % of predictions that are syntactically valid JSON
  - Schema conformance:  % that parse AND validate against MedicationList
  - Exact match:         % where prediction == gold (order-insensitive on list)
  - Field micro-F1:      per-field P/R/F across (drug, dose, route, frequency, duration)
  - Field macro-F1:      mean of per-field F1s

Pred vs gold matching: we treat each extraction as a set of
(drug_lower, field, value_lower) triples. This sidesteps the
list-alignment problem — a medication's dose is a hit iff the same
(drug, "dose", value) triple appears on both sides.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Sequence

from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).parent.parent))
from schema import MedicationList  # noqa: E402

FIELDS = ("drug", "dose", "route", "frequency", "duration")


@dataclass
class EvalResult:
    n: int
    parse_rate: float
    schema_conformance_rate: float
    exact_match_rate: float
    field_f1_micro: float
    field_f1_macro: float
    field_f1_per: dict = field(default_factory=dict)

    def to_row(self) -> dict:
        return {
            "n": self.n,
            "parse_rate": round(self.parse_rate, 4),
            "schema_conformance": round(self.schema_conformance_rate, 4),
            "exact_match": round(self.exact_match_rate, 4),
            "f1_micro": round(self.field_f1_micro, 4),
            "f1_macro": round(self.field_f1_macro, 4),
            **{f"f1_{f}": round(self.field_f1_per[f], 4) for f in FIELDS},
        }


def evaluate(predictions: Sequence[str], golds: Sequence[str]) -> EvalResult:
    assert len(predictions) == len(golds), "pred/gold length mismatch"
    n = len(predictions)

    parses = 0
    schema_ok = 0
    exact = 0

    tp = Counter()
    fp = Counter()
    fn = Counter()

    for pred_raw, gold_raw in zip(predictions, golds):
        pred_obj = _safe_parse(pred_raw)
        gold_obj = _safe_parse(gold_raw)

        if pred_obj is not None:
            parses += 1
            if _validates(pred_obj):
                schema_ok += 1

        gold_triples = _to_triples(gold_obj) if gold_obj is not None else set()
        pred_triples = _to_triples(pred_obj) if pred_obj is not None else set()

        if gold_triples == pred_triples:
            exact += 1

        for t in pred_triples & gold_triples:
            tp[t[1]] += 1
        for t in pred_triples - gold_triples:
            fp[t[1]] += 1
        for t in gold_triples - pred_triples:
            fn[t[1]] += 1

    per = {}
    for f in FIELDS:
        per[f] = _f1(tp[f], fp[f], fn[f])
    micro = _f1(sum(tp.values()), sum(fp.values()), sum(fn.values()))
    macro = sum(per.values()) / len(per)

    return EvalResult(
        n=n,
        parse_rate=parses / n,
        schema_conformance_rate=schema_ok / n,
        exact_match_rate=exact / n,
        field_f1_micro=micro,
        field_f1_macro=macro,
        field_f1_per=per,
    )


def _safe_parse(raw: str) -> dict | None:
    if raw is None:
        return None
    s = raw.strip()
    # models occasionally emit markdown fences despite instructions; strip them
    if s.startswith("```"):
        s = s.strip("`")
        s = s.partition("\n")[2] if "\n" in s else s
        if s.rstrip().endswith("```"):
            s = s.rstrip()[:-3]
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return None


def _validates(obj: dict) -> bool:
    try:
        MedicationList.model_validate(obj)
        return True
    except ValidationError:
        return False


def _to_triples(obj: dict) -> set[tuple[str, str, str]]:
    """Flatten into (drug_lower, field, value_lower) triples."""
    triples: set[tuple[str, str, str]] = set()
    meds = obj.get("medications") if isinstance(obj, dict) else None
    if not isinstance(meds, list):
        return triples
    for m in meds:
        if not isinstance(m, dict):
            continue
        drug = (m.get("drug") or "").strip().lower()
        if not drug:
            continue
        triples.add((drug, "drug", drug))
        for f in ("dose", "route", "frequency", "duration"):
            v = m.get(f)
            if v is None:
                continue
            triples.add((drug, f, str(v).strip().lower()))
    return triples


def _f1(tp: int, fp: int, fn: int) -> float:
    if tp == 0:
        return 0.0
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)
