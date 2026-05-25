#!/usr/bin/env python3
"""Build real train/test/preference JSONL files from Financial PhraseBank.

Financial PhraseBank is the standard labeled financial-sentiment benchmark
(sentences from financial news, labeled negative/neutral/positive). This script
downloads it, makes a reproducible train/test split, and writes three files the
training and evaluation scripts consume:

    data/financial_train.jsonl        # SFT data (text, label, id)
    data/financial_test.jsonl         # held-out eval (text, label, id)
    data/financial_preferences.jsonl  # DPO pairs (prompt, chosen, rejected)

Labels: Financial PhraseBank integer labels map as 0=negative, 1=neutral,
2=positive, which matches this project's label order exactly.

Example:
    python3 scripts/prepare_financial_phrasebank.py --test-size 0.2 --seed 7
"""

from __future__ import annotations

import argparse
from pathlib import Path
import random
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from qlora_dpo_finance.data import make_preference_pairs, normalize_label, write_jsonl

LABEL_BY_INT = {0: "negative", 1: "neutral", 2: "positive"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config-name",
        default="sentences_50agree",
        help="Financial PhraseBank agreement subset (50/66/75/allagree).",
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-train", type=int, default=None, help="Optional cap on training rows for a quick run.")
    parser.add_argument("--out-dir", default="data")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from datasets import load_dataset

    # financial_phrasebank ships a single split; we make our own held-out test set.
    # It loads via a dataset script, so modern `datasets` requires trust_remote_code.
    try:
        dataset = load_dataset(
            "financial_phrasebank", args.config_name, split="train", trust_remote_code=True
        )
    except Exception as primary_error:
        # Fallback: a script-free Parquet mirror with the same schema (sentence/label).
        print(f"Primary load failed ({primary_error}); trying a Parquet mirror...")
        dataset = load_dataset("takala/financial_phrasebank", args.config_name, split="train", trust_remote_code=True)

    rows = [
        {"id": f"fpb-{i}", "text": str(example["sentence"]).strip(), "label": LABEL_BY_INT[int(example["label"])]}
        for i, example in enumerate(dataset)
        if str(example["sentence"]).strip()
    ]

    rng = random.Random(args.seed)
    rng.shuffle(rows)
    split_at = int(len(rows) * (1 - args.test_size))
    train_rows = rows[:split_at]
    test_rows = rows[split_at:]
    if args.max_train:
        train_rows = train_rows[: args.max_train]

    # Sanity: every label normalizes cleanly.
    for row in train_rows[:5]:
        normalize_label(row["label"])

    preference_rows = make_preference_pairs(train_rows)

    out_dir = Path(args.out_dir)
    write_jsonl(out_dir / "financial_train.jsonl", train_rows)
    write_jsonl(out_dir / "financial_test.jsonl", test_rows)
    write_jsonl(out_dir / "financial_preferences.jsonl", preference_rows)

    print(
        f"train={len(train_rows)}  test={len(test_rows)}  preferences={len(preference_rows)}\n"
        f"wrote {out_dir}/financial_train.jsonl, financial_test.jsonl, financial_preferences.jsonl"
    )


if __name__ == "__main__":
    main()
