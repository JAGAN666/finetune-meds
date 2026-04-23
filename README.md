# Fine-Tuning Qwen2.5-7B for Medication Extraction → Structured JSON

End-to-end project that demonstrates why QLoRA fine-tuning beats prompting
for tightly-scoped structured extraction. Input: a free-text clinical
sentence. Output: strict JSON with `{drug, dose, route, frequency, duration}`
per medication.

The plan this was built from lives at
`~/.claude/plans/good-choice-this-is-quizzical-music.md`.

## Layout

```
.
├── schema.py                       # Pydantic model + JSON schema + system prompt
├── requirements.txt
├── data/
│   ├── prepare.py                  # MACCROBAT + BioLeaflets → nested-JSON parquet
│   ├── synthesize.py               # JSON-first synthetic generation
│   ├── train.parquet               # (generated) training mix
│   ├── val.parquet                 # (generated) train-time validation
│   ├── test.parquet                # (generated) held-out MACCROBAT docs
│   └── test_unseen_docs.parquet    # (generated) further-unseen MACCROBAT docs
├── src/
│   ├── format_chatml.py            # Qwen chat template wrapping
│   ├── train.py                    # Unsloth + TRL SFT loop
│   ├── eval.py                     # JSON/parse/exact/F1 metrics
│   ├── generate.py                 # Vanilla generation wrapper
│   └── generate_constrained.py     # Outlines-constrained generation
├── notebooks/
│   ├── 01_explore_dataset.ipynb    # EDA + audit gate + yield gate (run first)
│   ├── 02_train_qlora.ipynb        # Colab QLoRA training
│   └── 03_eval_and_compare.ipynb   # 3 runs × 2 test sets comparative eval
└── artifacts/                      # (generated) LoRA adapter + tokenizer
```

## How to run

1. **In Colab** (A100 recommended; T4 works at batch 1):
   - Open `notebooks/01_explore_dataset.ipynb`. Run every cell.
     - **Audit gate** (≥90% on 50 multi-drug sentences) must pass.
     - **Yield gate** sets the synthetic count and, if MACCROBAT yield
       is <1.5K, downgrades the target model to Qwen2.5-3B-Instruct.
     - Parquet files are written only if both gates pass.
   - Open `notebooks/02_train_qlora.ipynb`. Runs Unsloth QLoRA training,
     saves LoRA adapter to `artifacts/`.
   - Open `notebooks/03_eval_and_compare.ipynb`. Produces the
     (3 runs × 2 test sets) comparative table and a constrained-decoding
     variant.

2. **Locally (Mac)** is not supported for training (Unsloth is CUDA-only).
   Data prep and eval can run locally in a Python 3.10+ venv:
   ```bash
   python -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   ```

## Datasets

- **Primary — MACCROBAT_biomedical_ner** (`singh-aditya/MACCROBAT_biomedical_ner`)
  200 PMC clinical case reports, expert-annotated with ~20 biomedical entity
  types including Medication, Dosage, Frequency, Duration, Route. MIT license.
- **Supplementary — BioLeaflets** (`ruslan/bioleaflets-biomedical-ner`)
  1,336 EMA-authorised medicine package leaflets. Apache-2.0.
- **Synthetic** generated via Claude (budget ~$2–5).

### Optional stretch: n2c2 2018 Track 2

The n2c2 2018 Track 2 challenge corpus (discharge summaries with medication +
attribute annotations) is the gold standard for this task but requires a DUA.

To request access:
1. Visit https://portal.dbmi.hms.harvard.edu/ (Harvard DBMI Data Portal).
2. Register an account and sign the Data Use Agreement.
3. Request the **"2018 n2c2 Shared Task"** corpus (Track 2: ADE and Medication
   Extraction).
4. Turnaround is typically days.

If access arrives during the project, wire it in via the marked extension
point `_ingest_n2c2(...)` inside `data/prepare.py`. The surrounding pipeline
will pick it up without further changes.

## Citations

- Caufield JH, et al. *MACCROBAT-2018: A labeled corpus for clinical NLP.*
- Yermakov R, Drago N, Ziletti A. *BioLeaflets: A biomedical dataset for
  data2text generation.* (2021)
- 2018 n2c2 Shared Task, Harvard DBMI (if used).
