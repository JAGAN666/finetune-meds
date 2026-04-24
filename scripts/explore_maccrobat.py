"""Run with `%run scripts/explore_maccrobat.py` inside notebook 01.

Defines `docs`, `ent_counts`, `pool`, `test_docs`, `unseen_docs`, and
`pool_examples` in the notebook's namespace so subsequent cells can use
them without re-loading.
"""

import sys
import importlib
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import data.prepare
importlib.reload(data.prepare)
from data.prepare import load_maccrobat, partition_docs, doc_to_examples

docs = load_maccrobat()
print(f"Loaded {len(docs)} documents")

ent_counts = Counter()
for d in docs:
    for e in d.entities:
        ent_counts[e.kind] += 1
print("Entity counts (normalised):")
for k, v in ent_counts.most_common():
    print(f"  {k:10s} {v:>6}")

pool, test_docs, unseen_docs = partition_docs(docs)
print(f"Partition: pool={len(pool)}  test={len(test_docs)}  unseen={len(unseen_docs)}")

pool_examples = [ex for d in pool for ex in doc_to_examples(d)]
print(f"Pool sentence-level examples: {len(pool_examples)}")
