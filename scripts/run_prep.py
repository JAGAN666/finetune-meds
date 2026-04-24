"""Notebook 01 equivalent — one-shot data prep.

Runs the full flow without interactive gates:
  1. Load MACCROBAT, partition docs, entity-group.
  2. Print 10 random multi-drug samples for visual spot-check.
  3. Load BioLeaflets.
  4. Write train/val/test/test_unseen_docs parquets (no synthetics).
  5. Write artifacts/gate_decisions.json for notebook 02.

Audit and yield gates are auto-accepted with the 7B model target. Re-run
notebook 01 interactively if you want the real gating experience.

Usage (Colab or local):
    python scripts/run_prep.py
"""

from __future__ import annotations

import json
import random
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from data.prepare import (
    build_all,
    doc_to_examples,
    load_maccrobat,
    partition_docs,
)

random.seed(7)


def main() -> None:
    # ---- 1. Load + partition ----
    docs = load_maccrobat()
    print(f"Loaded {len(docs)} MACCROBAT documents")

    ent_counts = Counter(e.kind for d in docs for e in d.entities)
    print("Entity counts (normalised):")
    for k, v in ent_counts.most_common():
        print(f"  {k:10s} {v:>6}")

    pool, test_docs, unseen_docs = partition_docs(docs)
    print(f"Partition: pool={len(pool)}  test={len(test_docs)}  unseen={len(unseen_docs)}")

    pool_examples = [ex for d in pool for ex in doc_to_examples(d)]
    print(f"Pool sentence-level examples: {len(pool_examples)}")

    # ---- 2. Spot-check 10 multi-drug conversions ----
    multi = [
        ex for ex in pool_examples
        if len(json.loads(ex["target"])["medications"]) >= 2
    ]
    print(f"\n--- 10 multi-drug samples (spot-check) ---")
    for i, ex in enumerate(random.sample(multi, min(10, len(multi))), 1):
        print(f"[{i}] {ex['input'][:140]}")
        print(f"    -> {ex['target'][:180]}")

    # ---- 3. Build full parquet set (no synthetics for first pass) ----
    print("\nBuilding full training mix (no synthetics)...")
    summary = build_all(synthetic_examples=[])

    # ---- 4. Record gate decisions (auto: 7B, 0 synthetics) ----
    artifacts = PROJECT_ROOT / "artifacts"
    artifacts.mkdir(exist_ok=True)
    gate = {
        "audit_accuracy": 1.0,          # auto-accepted
        "maccrobat_pool_sentences": summary["train_maccrobat"],
        "synthetic_n": 0,
        "target_model": "Qwen/Qwen2.5-7B-Instruct",
        "note": "auto-run via scripts/run_prep.py — re-run notebook 01 for real gates",
    }
    (artifacts / "gate_decisions.json").write_text(json.dumps(gate, indent=2))
    print(f"\nGate decisions saved to artifacts/gate_decisions.json")
    print(json.dumps(gate, indent=2))

    # ---- 5. Integrity checks ----
    data_dir = PROJECT_ROOT / "data"
    train = pd.read_parquet(data_dir / "train.parquet")
    test = pd.read_parquet(data_dir / "test.parquet")
    unseen = pd.read_parquet(data_dir / "test_unseen_docs.parquet")

    train_ids = set(train["doc_id"])
    test_ids = set(test["doc_id"])
    unseen_ids = set(unseen["doc_id"])
    assert test_ids.isdisjoint(unseen_ids), "test / unseen-docs overlap"
    assert test_ids.isdisjoint(train_ids), "test leaked into train"
    assert unseen_ids.isdisjoint(train_ids), "unseen leaked into train"
    assert (test["source"] == "maccrobat").all(), "non-maccrobat in test"
    assert (unseen["source"] == "maccrobat").all(), "non-maccrobat in unseen"
    print("\nAll integrity checks passed.")


if __name__ == "__main__":
    main()
