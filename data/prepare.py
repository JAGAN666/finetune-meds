"""Dataset preparation pipeline.

Loads MACCROBAT + BioLeaflets from the Hugging Face Hub, reconstructs
sentences from token-level annotations, groups per-sentence entities into
our nested `{medications: [...]}` schema, partitions MACCROBAT at the
document level into training-pool / standard-test / unseen-docs-test, and
writes parquet files.

The entity-grouping heuristic (proximity-based assignment of
Dosage / Route / Frequency / Duration to the nearest Medication anchor) is
the most error-prone step in this pipeline. It must be hand-audited via
the Audit Gate in `notebooks/01_explore_dataset.ipynb` before any training
runs.

See also `synthesize.py` (synthetic augmentation) and the `_ingest_n2c2`
stub at the bottom of this file for the optional n2c2 2018 Track 2 source.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import pandas as pd
from datasets import load_dataset

DATA_DIR = Path(__file__).parent

# MACCROBAT entity names vary slightly across documents and conversions
# (e.g. some annotators used `Medication`, others `Drug`). Normalise to our
# five target fields. Unknown types are dropped during grouping.
ENTITY_ALIASES: dict[str, str] = {
    "medication": "drug",
    "drug": "drug",
    "dosage": "dose",
    "dose": "dose",
    "strength": "dose",
    "route": "route",
    "administration": "route",
    "frequency": "frequency",
    "duration": "duration",
    "form": None,  # kept out on purpose — "tablet" / "capsule" isn't in our schema
}

SENTENCE_TERMINATORS = {".", "?", "!", ";"}
CLAUSE_BOUNDARY_TOKENS = {",", ";", "then", "and", "also", "additionally", "subsequently"}


@dataclass
class Entity:
    start: int        # inclusive token index
    end: int          # exclusive token index
    kind: str         # normalised: drug | dose | route | frequency | duration
    text: str         # surface form (joined token text)


@dataclass
class Document:
    doc_id: str
    tokens: List[str]
    entities: List[Entity]


# ---------------------------------------------------------------------------
# MACCROBAT loading + parsing
# ---------------------------------------------------------------------------

def load_maccrobat() -> List[Document]:
    """Load MACCROBAT from HF and convert to internal Document objects.

    The HF dataset from `singh-aditya/MACCROBAT_biomedical_ner` exposes one
    row per document with fields `tokens`, `ner_tags` (BIO-encoded ints), and
    a `ner_labels` feature that maps tag int -> tag string. Exact field names
    are confirmed in notebook 01's EDA cell; if they differ, update the
    accessors below (kept narrow so drift is loud).
    """
    ds = load_dataset(
        "singh-aditya/MACCROBAT_biomedical_ner",
        split="train",
        trust_remote_code=True,
    )
    label_names = ds.features["ner_tags"].feature.names

    docs: List[Document] = []
    for idx, row in enumerate(ds):
        tokens = row["tokens"]
        tag_ids = row["ner_tags"]
        tag_strs = [label_names[t] for t in tag_ids]
        entities = _bio_to_entities(tokens, tag_strs)
        doc_id = row.get("id") or f"maccrobat_{idx:04d}"
        docs.append(Document(doc_id=str(doc_id), tokens=tokens, entities=entities))
    return docs


def _bio_to_entities(tokens: List[str], tags: List[str]) -> List[Entity]:
    entities: List[Entity] = []
    i = 0
    while i < len(tags):
        tag = tags[i]
        if tag == "O" or tag == "0":
            i += 1
            continue
        # BIO prefix handling: strip B-/I- to get the entity type
        prefix, _, kind_raw = tag.partition("-")
        if not kind_raw:  # dataset may store bare labels without B-/I-
            kind_raw = tag
        kind = ENTITY_ALIASES.get(kind_raw.lower())
        if kind is None:
            i += 1
            continue
        start = i
        i += 1
        while i < len(tags):
            nxt = tags[i]
            nxt_prefix, _, nxt_kind_raw = nxt.partition("-")
            nxt_kind = ENTITY_ALIASES.get((nxt_kind_raw or nxt).lower())
            if nxt_prefix == "I" and nxt_kind == kind:
                i += 1
                continue
            break
        end = i
        text = " ".join(tokens[start:end])
        entities.append(Entity(start=start, end=end, kind=kind, text=text))
    return entities


# ---------------------------------------------------------------------------
# Sentence splitting + entity grouping
# ---------------------------------------------------------------------------

def split_into_sentences(doc: Document) -> List[Tuple[int, int]]:
    """Return list of (start, end) token-index spans, one per sentence."""
    spans: List[Tuple[int, int]] = []
    start = 0
    for i, tok in enumerate(doc.tokens):
        if tok in SENTENCE_TERMINATORS:
            spans.append((start, i + 1))
            start = i + 1
    if start < len(doc.tokens):
        spans.append((start, len(doc.tokens)))
    return [s for s in spans if s[1] > s[0]]


def group_entities_in_sentence(
    sentence_tokens: List[str],
    entities: List[Entity],
    sent_start: int,
) -> dict:
    """Clause-aware proximity grouping: each Medication anchors a
    {drug, dose, route, frequency, duration} object; each non-drug entity
    attaches to the best drug by (token distance × clause-crossing penalty).

    Crossing a clause boundary (comma, semicolon, "then", "and", "also",
    "additionally", "subsequently") is penalised 10× so that
    "metoprolol ... for 2 weeks, then add lisinopril" correctly assigns
    "2 weeks" to metoprolol even though lisinopril is closer in raw
    token distance.

    If there are zero drugs: return {"medications": []}.
    """
    drugs = [e for e in entities if e.kind == "drug"]
    others = [e for e in entities if e.kind != "drug"]

    if not drugs:
        return {"medications": []}

    boundary_positions = [
        i for i, tok in enumerate(sentence_tokens)
        if tok.lower() in CLAUSE_BOUNDARY_TOKENS
    ]

    med_objs: List[dict] = []
    for d in drugs:
        med_objs.append({
            "drug": d.text,
            "dose": None,
            "route": None,
            "frequency": None,
            "duration": None,
            "_anchor_mid": (d.start + d.end) / 2,
        })

    for attr in others:
        mid = (attr.start + attr.end) / 2
        best_idx = min(
            range(len(med_objs)),
            key=lambda k: _assignment_cost(mid, med_objs[k]["_anchor_mid"], boundary_positions, k),
        )
        slot = attr.kind
        if med_objs[best_idx].get(slot) is None:
            med_objs[best_idx][slot] = attr.text

    for m in med_objs:
        m.pop("_anchor_mid", None)
    return {"medications": med_objs}


def _assignment_cost(attr_mid: float, drug_mid: float, boundaries: List[int], drug_idx: int) -> tuple:
    lo, hi = sorted([attr_mid, drug_mid])
    crossings = sum(1 for b in boundaries if lo < b < hi)
    distance = abs(attr_mid - drug_mid)
    # sort key: (penalised cost, stable tiebreaker on drug index)
    return (distance * (1 + 10 * crossings), drug_idx)


# ---------------------------------------------------------------------------
# Document partitioning
# ---------------------------------------------------------------------------

def partition_docs(
    docs: List[Document],
    pool_size: int = 160,
    test_size: int = 20,
    unseen_size: int = 20,
) -> Tuple[List[Document], List[Document], List[Document]]:
    """Deterministic doc-level partition via hashed IDs.

    Raises if the dataset has fewer docs than requested — notebook 01's
    yield gate catches this and rebalances.
    """
    if pool_size + test_size + unseen_size > len(docs):
        raise ValueError(
            f"Requested {pool_size+test_size+unseen_size} docs but dataset "
            f"has {len(docs)}. Adjust partition sizes."
        )
    ranked = sorted(
        docs,
        key=lambda d: hashlib.sha1(d.doc_id.encode()).hexdigest(),
    )
    pool = ranked[:pool_size]
    test = ranked[pool_size:pool_size + test_size]
    unseen = ranked[pool_size + test_size:pool_size + test_size + unseen_size]
    return pool, test, unseen


def doc_to_examples(doc: Document, keep_negative_fraction: float = 0.10) -> List[dict]:
    """Convert one document into sentence-level training examples.

    Each example is `{"input": <sentence>, "target": <medications-json>}`.
    Sentences with no drug entities are kept at `keep_negative_fraction` rate
    to prevent hallucinated extractions at inference time.
    """
    out: List[dict] = []
    neg_keep_counter = 0
    for sent_idx, (start, end) in enumerate(split_into_sentences(doc)):
        sent_toks = doc.tokens[start:end]
        sent_entities = [
            Entity(start=e.start - start, end=e.end - start, kind=e.kind, text=e.text)
            for e in doc.entities
            if e.start >= start and e.end <= end
        ]
        target = group_entities_in_sentence(sent_toks, sent_entities, start)
        if not target["medications"]:
            # keep roughly `keep_negative_fraction` of no-med sentences
            neg_keep_counter += 1
            if neg_keep_counter % int(1 / keep_negative_fraction) != 0:
                continue
        text = " ".join(sent_toks).strip()
        if not text:
            continue
        out.append({
            "input": text,
            "target": json.dumps(target, ensure_ascii=False),
            "source": "maccrobat",
            "doc_id": doc.doc_id,
        })
    return out


# ---------------------------------------------------------------------------
# BioLeaflets loading
# ---------------------------------------------------------------------------

POSOLOGY_MARKERS = re.compile(
    r"\b(dose|dosage|posology|how to take|administration|directions?)\b",
    re.IGNORECASE,
)


def load_bioleaflets(max_examples: int = 2000) -> List[dict]:
    """Load BioLeaflets, filter to posology sections, return sentence examples.

    The dataset doesn't carry medication-attribute annotations compatible with
    our schema, so we treat the leaflet text as a source of naturalistic
    patient-facing dosage language and re-annotate with a lightweight
    regex-based extractor for known dose / route / frequency patterns. This
    is NOISY by design — these examples exist for style coverage, not gold
    signal. Note the trade-off in notebook 01.
    """
    try:
        ds = load_dataset(
            "ruslan/bioleaflets-biomedical-ner",
            split="train",
            trust_remote_code=True,
        )
    except Exception as exc:
        print(f"[bioleaflets] load failed: {exc}; skipping")
        return []

    examples: List[dict] = []
    for row in ds:
        text_field = row.get("text") or row.get("document") or ""
        if not text_field or not POSOLOGY_MARKERS.search(text_field):
            continue
        for sent in _simple_sentence_split(text_field):
            target = _regex_extract_medication(sent)
            if target["medications"]:
                examples.append({
                    "input": sent,
                    "target": json.dumps(target, ensure_ascii=False),
                    "source": "bioleaflets",
                    "doc_id": str(row.get("id", "")),
                })
        if len(examples) >= max_examples:
            break
    return examples[:max_examples]


def _simple_sentence_split(text: str) -> List[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


_DOSE_RE = re.compile(r"\b\d+(?:\.\d+)?\s?(?:mg|g|mcg|ml|unit[s]?|iu|tablet[s]?|capsule[s]?)\b", re.I)
_FREQ_RE = re.compile(r"\b(?:once|twice|three times|four times|every\s+\d+\s+hours?|daily|bid|tid|qid|prn)\b", re.I)
_ROUTE_RE = re.compile(r"\b(?:by mouth|oral(?:ly)?|intravenous(?:ly)?|iv|po|subcutaneous(?:ly)?|sc|intramuscular(?:ly)?|im|topical(?:ly)?)\b", re.I)
_DURATION_RE = re.compile(r"\bfor\s+\d+\s+(?:day[s]?|week[s]?|month[s]?)\b", re.I)


def _regex_extract_medication(sentence: str) -> dict:
    """Very lightweight: flag a sentence as having a medication if it has a
    dose pattern; otherwise skip. Drug name is left as the longest
    capitalised word (weak heuristic — fine because this is augmentation
    signal, not gold test data)."""
    dose = _DOSE_RE.search(sentence)
    if not dose:
        return {"medications": []}
    freq = _FREQ_RE.search(sentence)
    route = _ROUTE_RE.search(sentence)
    duration = _DURATION_RE.search(sentence)
    caps = re.findall(r"\b[A-Z][a-z]{2,}\b", sentence)
    drug_name = max(caps, key=len) if caps else "unknown"
    return {"medications": [{
        "drug": drug_name,
        "dose": dose.group(0) if dose else None,
        "route": route.group(0) if route else None,
        "frequency": freq.group(0) if freq else None,
        "duration": duration.group(0) if duration else None,
    }]}


# ---------------------------------------------------------------------------
# n2c2 2018 Track 2 — extension point (OPTIONAL, requires DUA)
# ---------------------------------------------------------------------------

def _ingest_n2c2(n2c2_root: Optional[Path] = None) -> List[dict]:
    """Extension point for the n2c2 2018 Track 2 corpus.

    To wire this in:
    1. Obtain access via https://portal.dbmi.hms.harvard.edu/ (see README).
    2. Place the extracted `training-RiskFactors-Complete-Set1` and
       `test-RiskFactors-Complete-Set1` folders under `n2c2_root`.
    3. Implement the BRAT .ann + .txt parser below. The expected return
       format matches `doc_to_examples`:
       [{"input": <sentence>, "target": <json>, "source": "n2c2",
         "doc_id": <str>}, ...]
    4. Call this from `build_all()` and mix into the training pool.

    Left as a stub so the surrounding code is ready for a drop-in.
    """
    if n2c2_root is None:
        return []
    raise NotImplementedError(
        "n2c2 ingestion stub: implement BRAT parser when DUA access arrives."
    )


# ---------------------------------------------------------------------------
# Top-level orchestration
# ---------------------------------------------------------------------------

def build_all(
    synthetic_examples: Optional[List[dict]] = None,
    out_dir: Path = DATA_DIR,
    n2c2_root: Optional[Path] = None,
) -> dict:
    """Run the full prep pipeline and write parquet.

    `synthetic_examples` is passed in from the yield gate in notebook 01
    (it decides how many to generate based on MACCROBAT yield). Pass an
    empty list if augmentation is skipped.
    """
    print("Loading MACCROBAT...")
    docs = load_maccrobat()
    print(f"  got {len(docs)} documents")

    pool, test_docs, unseen_docs = partition_docs(docs)
    print(f"  partition: {len(pool)} pool, {len(test_docs)} test, {len(unseen_docs)} unseen")

    pool_examples: List[dict] = []
    for d in pool:
        pool_examples.extend(doc_to_examples(d))
    print(f"  pool sentences: {len(pool_examples)}")

    test_examples = [ex for d in test_docs for ex in doc_to_examples(d, keep_negative_fraction=1.0)]
    unseen_examples = [ex for d in unseen_docs for ex in doc_to_examples(d, keep_negative_fraction=1.0)]
    # keep only positives in test sets (evaluating hallucination separately via negatives in val)
    test_examples = [ex for ex in test_examples if json.loads(ex["target"])["medications"]]
    unseen_examples = [ex for ex in unseen_examples if json.loads(ex["target"])["medications"]]

    print("Loading BioLeaflets...")
    bioleaflets_examples = load_bioleaflets()
    print(f"  got {len(bioleaflets_examples)} BioLeaflets examples")

    n2c2_examples = _ingest_n2c2(n2c2_root) if n2c2_root else []

    # document-level train/val split inside the pool (~87.5/12.5)
    split_pivot = int(0.875 * len(pool))
    train_doc_ids = {d.doc_id for d in pool[:split_pivot]}
    train_pool = [ex for ex in pool_examples if ex["doc_id"] in train_doc_ids]
    val_pool = [ex for ex in pool_examples if ex["doc_id"] not in train_doc_ids]

    train_mix = train_pool + bioleaflets_examples + n2c2_examples + (synthetic_examples or [])

    out_dir.mkdir(parents=True, exist_ok=True)
    _write(out_dir / "train.parquet", train_mix)
    _write(out_dir / "val.parquet", val_pool)
    _write(out_dir / "test.parquet", test_examples)
    _write(out_dir / "test_unseen_docs.parquet", unseen_examples)

    summary = {
        "train": len(train_mix),
        "val": len(val_pool),
        "test": len(test_examples),
        "test_unseen_docs": len(unseen_examples),
        "train_maccrobat": len(train_pool),
        "train_bioleaflets": len(bioleaflets_examples),
        "train_synthetic": len(synthetic_examples or []),
        "train_n2c2": len(n2c2_examples),
    }
    print(f"Summary: {json.dumps(summary, indent=2)}")
    return summary


def _write(path: Path, examples: List[dict]) -> None:
    if not examples:
        print(f"  [warn] no examples for {path.name}, writing empty parquet")
    df = pd.DataFrame(examples)
    df.to_parquet(path, index=False)
    print(f"  wrote {path.name}: {len(df)} rows")


if __name__ == "__main__":
    build_all()
