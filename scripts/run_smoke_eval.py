#!/usr/bin/env python3
"""Run local metric evaluation on JSONL predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from qlora_dpo_finance.data import read_jsonl
from qlora_dpo_finance.metrics import evaluate_predictions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--baseline", default="data/samples/baseline_sample.jsonl")
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = evaluate_predictions(read_jsonl(args.predictions), read_jsonl(args.baseline))
    rendered = json.dumps(metrics, indent=2)
    print(rendered)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
