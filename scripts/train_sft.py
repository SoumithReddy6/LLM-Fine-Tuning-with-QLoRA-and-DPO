#!/usr/bin/env python3
"""Run QLoRA supervised fine-tuning for financial sentiment."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from qlora_dpo_finance.config import load_config
from qlora_dpo_finance.data import load_hf_financial_dataset, read_jsonl, to_sft_records
from qlora_dpo_finance.trainer import build_lora_config, load_model_and_tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--local-data", default=None, help="Optional JSONL override for smoke-scale SFT data.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    dataset_config = config.section("dataset")
    training_config = config.section("training")

    if args.local_data:
        train_records = to_sft_records(read_jsonl(args.local_data))
    else:
        dataset = load_hf_financial_dataset(dataset_config["source"], split="train")
        train_records = to_sft_records(
            dataset.select(range(min(len(dataset), int(dataset_config.get("max_train_samples", len(dataset)))))),
            text_field=dataset_config.get("text_field", "sentence"),
            label_field=dataset_config.get("label_field", "label"),
        )

    from datasets import Dataset
    from peft import get_peft_model, prepare_model_for_kbit_training
    from transformers import TrainingArguments
    from trl import SFTTrainer

    model, tokenizer = load_model_and_tokenizer(config)
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, build_lora_config(config.section("lora")))

    train_dataset = Dataset.from_list(train_records)
    args_out = TrainingArguments(
        output_dir=training_config["output_dir"],
        num_train_epochs=float(training_config.get("num_train_epochs", 2)),
        per_device_train_batch_size=int(training_config.get("per_device_train_batch_size", 2)),
        gradient_accumulation_steps=int(training_config.get("gradient_accumulation_steps", 8)),
        learning_rate=float(training_config.get("learning_rate", 2e-4)),
        warmup_ratio=float(training_config.get("warmup_ratio", 0.03)),
        logging_steps=int(training_config.get("logging_steps", 10)),
        save_steps=int(training_config.get("save_steps", 250)),
        bf16=bool(training_config.get("bf16", True)),
        report_to=["mlflow", "wandb"],
    )
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        dataset_text_field="text",
        max_seq_length=int(training_config.get("max_seq_length", 1024)),
        args=args_out,
    )
    trainer.train()
    trainer.save_model(training_config["output_dir"])


if __name__ == "__main__":
    main()
