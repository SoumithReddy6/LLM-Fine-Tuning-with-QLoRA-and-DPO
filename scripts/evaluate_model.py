#!/usr/bin/env python3
"""Evaluate model prediction files and log metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from qlora_dpo_finance.config import load_config
from qlora_dpo_finance.data import read_jsonl
from qlora_dpo_finance.metrics import evaluate_predictions
from qlora_dpo_finance.tracking import log_to_mlflow


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def main() -> None:
    config = load_config(parse_args().config)
    evaluation = config.section("evaluation")
    metrics = evaluate_predictions(
        read_jsonl(evaluation["predictions_path"]),
        read_jsonl(evaluation["baseline_path"]) if evaluation.get("baseline_path") else None,
    )
    output_path = Path(evaluation.get("output_path", "artifacts/evaluation_metrics.json"))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(metrics, indent=2)
    output_path.write_text(rendered, encoding="utf-8")
    print(rendered)
    tracking = config.section("tracking")
    log_to_mlflow(tracking.get("mlflow_experiment", config.name), config.raw, metrics)


if __name__ == "__main__":
    main()
