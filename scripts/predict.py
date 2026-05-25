#!/usr/bin/env python3
"""Run a causal LM (optionally with a trained LoRA adapter) on financial
sentiment data and write a predictions JSONL.

This is the missing bridge between training and metrics. ``evaluate_model.py``
only scores an existing predictions file; this script *produces* that file from
a real model, so reported numbers come from inference rather than hand-authored
rows.

Run it twice to measure the lift fine-tuning actually bought:

    # Baseline: base model, zero-shot (no adapter)
    python3 scripts/predict.py --base-model Qwen/Qwen2.5-0.5B-Instruct \
        --input data/financial_test.jsonl --output artifacts/baseline_preds.jsonl --load-in-4bit

    # Fine-tuned: same base model + the adapter you trained
    python3 scripts/predict.py --base-model Qwen/Qwen2.5-0.5B-Instruct \
        --adapter outputs/qwen25_0_5b_qlora_rank16 \
        --input data/financial_test.jsonl --output artifacts/tuned_preds.jsonl --load-in-4bit
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from qlora_dpo_finance.data import format_prompt, normalize_label, read_jsonl, write_jsonl
from qlora_dpo_finance.metrics import parse_prediction


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", required=True, help="HF model id or local path for the base model.")
    parser.add_argument("--adapter", default=None, help="Path to a trained LoRA adapter. Omit for the zero-shot baseline.")
    parser.add_argument("--input", required=True, help="JSONL eval file with `text` (or `sentence`) and `label` fields.")
    parser.add_argument("--output", required=True, help="Where to write the predictions JSONL.")
    parser.add_argument("--max-new-tokens", type=int, default=5)
    parser.add_argument("--load-in-4bit", action="store_true", help="Load the base model in 4-bit (QLoRA-style).")
    parser.add_argument("--limit", type=int, default=None, help="Only score the first N rows (useful for a quick check).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: dict = {"trust_remote_code": True, "device_map": "auto"}
    if args.load_in_4bit:
        from transformers import BitsAndBytesConfig

        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    else:
        model_kwargs["torch_dtype"] = torch.bfloat16

    model = AutoModelForCausalLM.from_pretrained(args.base_model, **model_kwargs)
    if args.adapter:
        # Fail clearly if the adapter folder is missing locally. Otherwise PEFT
        # tries to fetch it from the Hugging Face Hub and raises a confusing 401.
        adapter_dir = Path(args.adapter)
        if not (adapter_dir / "adapter_config.json").exists():
            raise FileNotFoundError(
                f"No trained adapter at '{args.adapter}'. Run the training step that "
                f"produces it before predicting (e.g. scripts/train_sft.py or "
                f"scripts/train_dpo.py). Looked for {adapter_dir / 'adapter_config.json'}."
            )
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    rows = read_jsonl(args.input)
    if args.limit:
        rows = rows[: args.limit]

    outputs: list[dict] = []
    for index, row in enumerate(rows):
        text = str(row.get("text") or row.get("sentence") or "").strip()
        gold = normalize_label(row["label"])
        prompt = format_prompt(text)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        started = time.perf_counter()
        with torch.no_grad():
            generated = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                return_dict_in_generate=True,
                output_scores=True,
                pad_token_id=tokenizer.pad_token_id,
            )
        latency_ms = round((time.perf_counter() - started) * 1000, 2)

        new_token_ids = generated.sequences[0][inputs["input_ids"].shape[1] :]
        completion = tokenizer.decode(new_token_ids, skip_special_tokens=True).strip()

        # Confidence = mean probability the model assigned to the tokens it chose.
        token_probs: list[float] = []
        for token_id, step_scores in zip(new_token_ids, generated.scores):
            probability = torch.softmax(step_scores[0], dim=-1)[token_id].item()
            token_probs.append(probability)
        confidence = round(sum(token_probs) / len(token_probs), 4) if token_probs else 0.0

        # Store the raw completion so evaluate_model.py can measure invalid-label
        # and hallucination-proxy rates faithfully (it re-parses this field).
        outputs.append(
            {
                "id": row.get("id", f"ex-{index}"),
                "label": gold,
                "prediction": parse_prediction(completion) or completion,
                "confidence": confidence,
                "latency_ms": latency_ms,
            }
        )

    write_jsonl(args.output, outputs)
    print(f"Wrote {len(outputs)} predictions to {args.output}")


if __name__ == "__main__":
    main()
