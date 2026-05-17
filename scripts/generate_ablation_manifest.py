#!/usr/bin/env python3
"""Generate a CSV manifest for QLoRA/DPO ablation runs."""

from __future__ import annotations

import argparse
import csv
from itertools import product
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", default="configs/sweeps/ablation_grid.yaml")
    parser.add_argument("--output", default="artifacts/ablation_manifest.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    grid = yaml.safe_load(Path(args.grid).read_text(encoding="utf-8"))
    fields = ["run_id", "model", "learning_rate", "lora_rank", "data_mixture"]
    rows = []
    for index, values in enumerate(
        product(grid["models"], grid["learning_rates"], grid["lora_ranks"], grid["data_mixtures"]),
        start=1,
    ):
        model, learning_rate, lora_rank, data_mixture = values
        rows.append(
            {
                "run_id": f"ablation-{index:03d}",
                "model": model,
                "learning_rate": learning_rate,
                "lora_rank": lora_rank,
                "data_mixture": data_mixture,
            }
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} ablation runs to {output_path}")


if __name__ == "__main__":
    main()
