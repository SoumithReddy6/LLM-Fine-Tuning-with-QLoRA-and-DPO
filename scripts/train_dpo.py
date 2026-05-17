#!/usr/bin/env python3
"""Run DPO preference tuning from financial sentiment pairs."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from qlora_dpo_finance.config import load_config
from qlora_dpo_finance.data import read_jsonl
from qlora_dpo_finance.trainer import build_lora_config, load_model_and_tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def main() -> None:
    config = load_config(parse_args().config)
    dataset_config = config.section("dataset")
    training_config = config.section("training")

    from datasets import Dataset
    from peft import PeftModel, get_peft_model, prepare_model_for_kbit_training
    from transformers import TrainingArguments
    from trl import DPOTrainer

    model, tokenizer = load_model_and_tokenizer(config)
    model = prepare_model_for_kbit_training(model)
    adapter_path = config.section("model").get("sft_adapter_path")
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=True)
    else:
        model = get_peft_model(model, build_lora_config(config.section("lora")))

    preference_dataset = Dataset.from_list(read_jsonl(dataset_config["preference_path"]))
    args_out = TrainingArguments(
        output_dir=training_config["output_dir"],
        num_train_epochs=float(training_config.get("num_train_epochs", 1)),
        per_device_train_batch_size=int(training_config.get("per_device_train_batch_size", 1)),
        gradient_accumulation_steps=int(training_config.get("gradient_accumulation_steps", 8)),
        learning_rate=float(training_config.get("learning_rate", 5e-5)),
        logging_steps=int(training_config.get("logging_steps", 10)),
        save_steps=int(training_config.get("save_steps", 250)),
        bf16=bool(training_config.get("bf16", True)),
        report_to=["mlflow", "wandb"],
    )
    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=args_out,
        beta=float(training_config.get("beta", 0.1)),
        train_dataset=preference_dataset,
        tokenizer=tokenizer,
        max_prompt_length=int(training_config.get("max_prompt_length", 768)),
        max_length=int(training_config.get("max_length", 1024)),
    )
    trainer.train()
    trainer.save_model(training_config["output_dir"])


if __name__ == "__main__":
    main()
