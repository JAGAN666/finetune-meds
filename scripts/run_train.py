"""Notebook 02 equivalent — one-shot QLoRA training.

Reads artifacts/gate_decisions.json for the target model, adapts batch
size to the detected GPU (A100 = bigger, T4 = smaller), and trains.

Prerequisites:
    - Run `scripts/run_prep.py` first to produce data/*.parquet and
      artifacts/gate_decisions.json.
    - Install Unsloth (see README / notebook 02).

Usage:
    python scripts/run_train.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main() -> None:
    import torch
    from src.train import TrainConfig, train

    gate_path = PROJECT_ROOT / "artifacts" / "gate_decisions.json"
    if not gate_path.exists():
        raise FileNotFoundError(
            f"{gate_path} not found. Run scripts/run_prep.py first."
        )
    gate = json.loads(gate_path.read_text())
    print("Gate decisions:")
    print(json.dumps(gate, indent=2))

    if not torch.cuda.is_available():
        raise RuntimeError(
            "No CUDA GPU detected. Training must run on Colab (Runtime -> "
            "Change runtime type -> T4 or A100)."
        )
    gpu_name = torch.cuda.get_device_name(0)
    print(f"GPU: {gpu_name}")
    is_a100 = "A100" in gpu_name

    model_name = gate["target_model"]
    slug = model_name.split("/")[-1].lower().replace(".", "")
    output_dir = PROJECT_ROOT / "artifacts" / f"{slug}-meds-qlora"

    cfg = TrainConfig(
        model_name=model_name,
        per_device_train_batch_size=2 if is_a100 else 1,
        gradient_accumulation_steps=4 if is_a100 else 8,
        output_dir=output_dir,
    )
    print(f"Output dir: {cfg.output_dir}")
    train(cfg)
    print(f"\nTraining complete. Adapter saved at {cfg.output_dir}")


if __name__ == "__main__":
    main()
