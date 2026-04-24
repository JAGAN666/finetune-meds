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
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import pandas as pd

DATA_DIR = Path(__file__).parent
CACHE_DIR = DATA_DIR / ".cache"
MACCROBAT_URL = (
    "https://huggingface.co/datasets/singh-aditya/MACCROBAT_biomedical_ner/"
    "resolve/main/MACCROBAT2020-V2.json"
)

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

SENTENCE_TERMINATOR_RE = re.compile(r"(?<=[.!?;])\s+")
# clause boundaries inside a sentence — used to penalise attribute-to-drug
# assignments that cross a clause. The `,` + optional connector pattern
# catches "... for 2 weeks, then add ...". Bare `;` is also a boundary.
CLAUSE_BOUNDARY_RE = re.compile(
    r",\s*(?:then|and|also|additionally|subsequently)\b|;|\n",
    re.IGNORECASE,
)


@dataclass
class Entity:
    start: int        # character offset into `full_text`, inclusive
    end: int          # character offset, exclusive
    kind: str         # normalised: drug | dose | route | frequency | duration
    text: str         # surface form


@dataclass
class Document:
    doc_id: str
    full_text: str
    entities: List[Entity]


# ---------------------------------------------------------------------------
# MACCROBAT loading + parsing
# ---------------------------------------------------------------------------

def _download_with_ssl_fallback(url: str, dest: Path) -> None:
    """Download `url` -> `dest`. Falls back to `certifi`'s bundle on Macs
    where the system Python doesn't have root certs configured.
    """
    import ssl
    try:
        urllib.request.urlretrieve(url, dest)
        return
    except (urllib.error.URLError, ssl.SSLError) as exc:
        if "CERTIFICATE_VERIFY_FAILED" not in str(exc):
            raise
    # SSL verify failed — retry with certifi if available
    try:
        import certifi
    except ImportError as e:
        raise RuntimeError(
            "SSL cert verification failed and `certifi` isn't installed. "
            "Run `pip install certifi` or install Mac's Python root certs via "
            "`/Applications/Python\\ 3.X/Install\\ Certificates.command`."
        ) from e
    ctx = ssl.create_default_context(cafile=certifi.where())
    opener = urllib.request.build_opener(urllib.request.HTTPSHandler(context=ctx))
    with opener.open(url) as resp, open(dest, "wb") as f:
        while chunk := resp.read(1 << 16):
            f.write(chunk)


def load_maccrobat() -> List[Document]:
    """Load MACCROBAT by fetching the raw JSON from Hugging Face directly.

    We bypass `datasets.load_dataset` on purpose — the repo ships a loading
    script which `datasets>=3.0` refuses to execute. The raw JSON
    (`MACCROBAT2020-V2.json`, ~8 MB) is public and self-contained, so a
    plain HTTPS download is both simpler and version-agnostic.

    The file's shape:
      {
        "data": [{"full_text": str, "ner_info": [{text, label, start, end}, ...],
                  "tokens": [...], "ner_labels": [...]}, ...],
        "all_ner_labels": [...],
        ...
      }

    We use `ner_info` directly (char-offset spans on `full_text`), which
    dodges the tokenisation quirks of the `tokens` field (which includes
    whitespace-only tokens).
    """
    CACHE_DIR.mkdir(exist_ok=True)
    cached = CACHE_DIR / "MACCROBAT2020-V2.json"
    if not cached.exists():
        print(f"Downloading {MACCROBAT_URL} -> {cached}")
        _download_with_ssl_fallback(MACCROBAT_URL, cached)

    with open(cached) as f:
        raw = json.load(f)

    docs: List[Document] = []
    for idx, item in enumerate(raw["data"]):
        full_text = item["full_text"]
        entities: List[Entity] = []
        for info in item["ner_info"]:
            kind = ENTITY_ALIASES.get(info["label"].lower())
            if kind is None:
                continue
            entities.append(Entity(
                start=info["start"],
                end=info["end"],
                kind=kind,
                text=info["text"],
            ))
        doc_id = f"maccrobat_{idx:04d}"
        docs.append(Document(doc_id=doc_id, full_text=full_text, entities=entities))
    return docs


# ---------------------------------------------------------------------------
# Sentence splitting + entity grouping
# ---------------------------------------------------------------------------

def split_into_sentences(doc: Document) -> List[Tuple[int, int]]:
    """Return (char_start, char_end) spans of each sentence in full_text."""
    text = doc.full_text
    spans: List[Tuple[int, int]] = []
    start = 0
    for m in SENTENCE_TERMINATOR_RE.finditer(text):
        end = m.start()
        if end > start:
            spans.append((start, end))
        start = m.end()
    if start < len(text):
        spans.append((start, len(text)))
    return spans


def group_entities_in_sentence(
    sentence_text: str,
    entities: List[Entity],
    sent_start: int,
) -> dict:
    """Clause-aware proximity grouping on character offsets.

    Each Medication anchors a {drug, dose, route, frequency, duration}
    object; each non-drug entity attaches to the best drug by
    (char distance x clause-crossing penalty). Crossing a clause boundary
    (comma followed by then/and/also/additionally/subsequently; or ';'
    or a newline) is penalised 10x so that
    "metoprolol ... for 2 weeks, then add lisinopril" correctly assigns
    "2 weeks" to metoprolol even though lisinopril is closer in raw chars.
    """
    drugs = [e for e in entities if e.kind == "drug"]
    others = [e for e in entities if e.kind != "drug"]

    if not drugs:
        return {"medications": []}

    # clause-boundary char offsets *relative to full_text* — same frame as
    # Entity.start/end, so we can compare directly.
    boundary_positions = [
        sent_start + m.start() for m in CLAUSE_BOUNDARY_RE.finditer(sentence_text)
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
        sent_text = doc.full_text[start:end].strip()
        if not sent_text:
            continue
        sent_entities = [
            e for e in doc.entities if e.start >= start and e.end <= end
        ]
        target = group_entities_in_sentence(sent_text, sent_entities, start)
        if not target["medications"]:
            neg_keep_counter += 1
            if neg_keep_counter % int(1 / keep_negative_fraction) != 0:
                continue
        out.append({
            "input": sent_text,
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
    """Load BioLeaflets, extract sentence-level posology examples.

    Schema observed: `{ID, URL, Product_Name, Full_Content, Section_1..6}`
    where Section_3 is consistently the "how to use" / posology section and
    Section_2 covers warnings+interactions. We use Section_3 as the posology
    source; it's stringified Python dicts we have to literal_eval.

    The dataset doesn't carry medication-attribute annotations compatible
    with our schema, so we re-annotate with a lightweight regex extractor.
    The important advantage over pure regex: we know `Product_Name` for each
    leaflet, so the "drug" field is the actual product, not a
    longest-capitalised-word guess. These examples are NOISY but provide
    patient-facing posology style that MACCROBAT's clinical prose doesn't.
    """
    import ast
    try:
        from datasets import load_dataset
        ds = load_dataset("ruslan/bioleaflets-biomedical-ner", split="train")
    except Exception as exc:
        print(f"[bioleaflets] load failed: {exc}; skipping")
        return []

    examples: List[dict] = []
    for row in ds:
        product = row.get("Product_Name") or ""
        section_raw = row.get("Section_3") or ""
        if not section_raw:
            continue
        try:
            section = ast.literal_eval(section_raw)
            content = section.get("Section_Content", "") if isinstance(section, dict) else ""
        except (SyntaxError, ValueError):
            continue
        if not content:
            continue
        for sent in _simple_sentence_split(content):
            target = _regex_extract_medication(sent, product_name=product)
            if target["medications"]:
                examples.append({
                    "input": sent,
                    "target": json.dumps(target, ensure_ascii=False),
                    "source": "bioleaflets",
                    "doc_id": str(row.get("ID", "")),
                })
        if len(examples) >= max_examples:
            break
    return examples[:max_examples]


def _simple_sentence_split(text: str) -> List[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


_DOSE_RE = re.compile(r"\b\d+(?:\.\d+)?\s?(?:mg|g|mcg|ml|unit[s]?|iu|tablet[s]?|capsule[s]?)\b", re.I)
_FREQ_RE = re.compile(r"\b(?:once|twice|three times|four times|every\s+\d+\s+hours?|daily|bid|tid|qid|prn)\b", re.I)
_ROUTE_RE = re.compile(r"\b(?:by mouth|oral(?:ly)?|intravenous(?:ly)?|iv|po|subcutaneous(?:ly)?|sc|intramuscular(?:ly)?|im|topical(?:ly)?|injection)\b", re.I)
_DURATION_RE = re.compile(r"\bfor\s+\d+\s+(?:day[s]?|week[s]?|month[s]?)\b", re.I)


def _regex_extract_medication(sentence: str, product_name: str = "") -> dict:
    """Return a single-drug extraction if the sentence contains a dose
    pattern AND a drug name that actually appears in the sentence.

    Drug-name selection (in order):
      1. `product_name` if it appears in the sentence (case-insensitive).
      2. The longest capitalised word from the sentence itself (>=3 chars).
      3. Otherwise skip — training on a drug name absent from the input
         would teach the model to hallucinate.
    """
    dose = _DOSE_RE.search(sentence)
    if not dose:
        return {"medications": []}

    drug_name = ""
    pn = product_name.strip()
    if pn and pn.lower() in sentence.lower():
        # match case as found in the sentence for fidelity
        idx = sentence.lower().find(pn.lower())
        drug_name = sentence[idx:idx + len(pn)]
    else:
        caps = re.findall(r"\b[A-Z][a-z]{2,}\b", sentence)
        if caps:
            drug_name = max(caps, key=len)
    if not drug_name:
        return {"medications": []}

    freq = _FREQ_RE.search(sentence)
    route = _ROUTE_RE.search(sentence)
    duration = _DURATION_RE.search(sentence)
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
