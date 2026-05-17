"""Dataset normalization and prompt-building utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable


LABELS = ("negative", "neutral", "positive")


def read_jsonl(path: str | Path) -> list[dict]:
    source = Path(path)
    return [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(json.dumps(row, ensure_ascii=True) for row in rows) + "\n", encoding="utf-8")


def normalize_label(value: str | int) -> str:
    if isinstance(value, int):
        return LABELS[value]
    label = str(value).strip().lower()
    if label in LABELS:
        return label
    aliases = {"bearish": "negative", "mixed": "neutral", "bullish": "positive"}
    if label in aliases:
        return aliases[label]
    raise ValueError(f"Unsupported label: {value}")


def format_prompt(text: str) -> str:
    return (
        "Classify the financial sentiment as negative, neutral, or positive.\n"
        f"Text: {text.strip()}\n"
        "Label:"
    )


def to_sft_records(rows: Iterable[dict], text_field: str = "text", label_field: str = "label") -> list[dict]:
    records: list[dict] = []
    for row in rows:
        text = str(row.get(text_field) or row.get("sentence") or row.get("text") or "").strip()
        if not text:
            continue
        label = normalize_label(row[label_field])
        records.append(
            {
                "id": row.get("id"),
                "prompt": format_prompt(text),
                "response": label,
                "text": f"{format_prompt(text)} {label}",
                "label": label,
            }
        )
    return records


def make_preference_pairs(rows: Iterable[dict], text_field: str = "text", label_field: str = "label") -> list[dict]:
    pairs: list[dict] = []
    for row in rows:
        text = str(row.get(text_field) or row.get("sentence") or row.get("text") or "").strip()
        if not text:
            continue
        chosen = normalize_label(row[label_field])
        rejected = next(label for label in LABELS if label != chosen)
        pairs.append(
            {
                "id": row.get("id"),
                "prompt": format_prompt(text),
                "chosen": chosen,
                "rejected": rejected,
            }
        )
    return pairs


def load_hf_financial_dataset(dataset_name: str, split: str = "train"):
    try:
        from datasets import load_dataset
    except Exception as exc:
        raise RuntimeError("Install datasets to load Hugging Face datasets") from exc
    return load_dataset(dataset_name, split=split)
