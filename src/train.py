"""Unsloth + TRL QLoRA trainer for Qwen2.5-7B-Instruct.

Callable from notebook 02 (Colab) or the CLI. The model name is a parameter
because the yield gate in notebook 01 may downgrade to Qwen2.5-3B if the
MACCROBAT training pool yields fewer than 1.5K sentences.

All hyperparameters are dataclass-defaulted to the values in the plan.
Override via CLI flags or by passing a custom TrainConfig.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.format_chatml import format_for_training  # noqa: E402


@dataclass
class TrainConfig:
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    max_seq_len: int = 1024
    load_in_4bit: bool = True
    # LoRA
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    # optim
    learning_rate: float = 2e-4
    num_train_epochs: int = 3
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    optim: str = "paged_adamw_8bit"
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    # IO
    train_path: Path = Path(__file__).parent.parent / "data" / "train.parquet"
    val_path: Path = Path(__file__).parent.parent / "data" / "val.parquet"
    output_dir: Path = Path(__file__).parent.parent / "artifacts" / "qwen2.5-7b-meds-qlora"
    logging_steps: int = 10
    eval_steps: int = 100
    save_steps: int = 500
    seed: int = 42


def train(cfg: Optional[TrainConfig] = None):
    cfg = cfg or TrainConfig()
    # Unsloth imports are heavy and CUDA-only; keep them inside the function
    # so `import train` works on machines without a GPU (for tests / linting).
    from unsloth import FastLanguageModel
    from datasets import load_dataset
    from trl import SFTConfig, SFTTrainer

    print(f"Loading {cfg.model_name} in 4-bit...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg.model_name,
        max_seq_length=cfg.max_seq_len,
        load_in_4bit=cfg.load_in_4bit,
        dtype=None,  # auto
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=cfg.seed,
    )

    print("Loading datasets...")
    ds = load_dataset(
        "parquet",
        data_files={"train": str(cfg.train_path), "val": str(cfg.val_path)},
    )
    # Build the chat-templated `text`, then pre-tokenize into
    # input_ids/attention_mask. Pre-tokenising is more robust than relying
    # on the TRL SFTTrainer's on-the-fly tokenisation path, which can
    # collide with the default DataCollatorForLanguageModeling when string
    # columns survive into collation.
    raw_cols = ds["train"].column_names

    def _prep(ex):
        text = format_for_training(ex, tokenizer)["text"]
        enc = tokenizer(
            text,
            truncation=True,
            max_length=cfg.max_seq_len,
            add_special_tokens=False,
        )
        enc["labels"] = enc["input_ids"].copy()
        return enc

    ds = ds.map(_prep, remove_columns=raw_cols)
    print(f"  train: {len(ds['train'])} examples; val: {len(ds['val'])} examples")
    print(f"  columns: {ds['train'].column_names}")

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    args = SFTConfig(
        output_dir=str(cfg.output_dir),
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        num_train_epochs=cfg.num_train_epochs,
        learning_rate=cfg.learning_rate,
        lr_scheduler_type=cfg.lr_scheduler_type,
        warmup_ratio=cfg.warmup_ratio,
        optim=cfg.optim,
        logging_steps=cfg.logging_steps,
        eval_strategy="steps",
        eval_steps=cfg.eval_steps,
        save_steps=cfg.save_steps,
        save_total_limit=3,
        report_to="none",
        seed=cfg.seed,
        packing=False,
    )

    # Pre-tokenised dataset -> pad with a language-model collator.
    from transformers import DataCollatorForLanguageModeling
    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=ds["train"],
        eval_dataset=ds["val"],
        args=args,
        data_collator=collator,
    )
    trainer.train()

    trainer.save_model(str(cfg.output_dir))
    tokenizer.save_pretrained(str(cfg.output_dir))
    print(f"Saved adapter + tokenizer to {cfg.output_dir}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=4)
    args = p.parse_args()
    cfg = TrainConfig(
        model_name=args.model,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
    )
    train(cfg)
