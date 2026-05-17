#!/usr/bin/env python3
"""Create DPO preference pairs from labeled financial sentiment records."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from qlora_dpo_finance.data import make_preference_pairs, read_jsonl, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/samples/financial_sentiment_sample.jsonl")
    parser.add_argument("--output", default="artifacts/preferences.jsonl")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pairs = make_preference_pairs(read_jsonl(args.input))
    write_jsonl(args.output, pairs)
    print(f"Wrote {len(pairs)} preference pairs to {args.output}")


if __name__ == "__main__":
    main()
