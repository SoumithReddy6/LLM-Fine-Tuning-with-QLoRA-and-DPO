#!/usr/bin/env python3
"""Run a zero-shot OpenAI baseline for financial sentiment."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from qlora_dpo_finance.data import format_prompt, read_jsonl, write_jsonl
from qlora_dpo_finance.metrics import parse_prediction


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/samples/financial_sentiment_sample.jsonl")
    parser.add_argument("--output", default="artifacts/openai_baseline.jsonl")
    parser.add_argument("--model", default="gpt-4o")
    return parser.parse_args()


def main() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Set OPENAI_API_KEY before running the OpenAI baseline")

    from openai import OpenAI

    args = parse_args()
    client = OpenAI()
    outputs = []
    for row in read_jsonl(args.input):
        started = time.perf_counter()
        response = client.chat.completions.create(
            model=args.model,
            temperature=0,
            messages=[
                {"role": "system", "content": "Return exactly one label: negative, neutral, or positive."},
                {"role": "user", "content": format_prompt(row["text"])},
            ],
        )
        text = response.choices[0].message.content or ""
        outputs.append(
            {
                "id": row["id"],
                "label": row["label"],
                "prediction": parse_prediction(text) or text.strip(),
                "confidence": 1.0,
                "latency_ms": round((time.perf_counter() - started) * 1000, 2),
            }
        )
    write_jsonl(args.output, outputs)
    print(f"Wrote {len(outputs)} baseline predictions to {args.output}")


if __name__ == "__main__":
    main()
