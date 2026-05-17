"""Evaluation metrics for financial sentiment classification."""

from __future__ import annotations

from collections import Counter
from math import isfinite
from typing import Iterable

from qlora_dpo_finance.data import LABELS


def parse_prediction(value: str) -> str | None:
    normalized = str(value).strip().lower()
    tokens = [label for label in LABELS if label in normalized.split()]
    if normalized in LABELS:
        return normalized
    if len(tokens) == 1:
        return tokens[0]
    return None


def accuracy_score(labels: list[str], predictions: list[str | None]) -> float:
    if not labels:
        return 0.0
    correct = sum(1 for label, prediction in zip(labels, predictions) if prediction == label)
    return correct / len(labels)


def macro_f1_score(labels: list[str], predictions: list[str | None]) -> float:
    scores: list[float] = []
    for label in LABELS:
        true_positive = sum(1 for gold, pred in zip(labels, predictions) if gold == label and pred == label)
        false_positive = sum(1 for gold, pred in zip(labels, predictions) if gold != label and pred == label)
        false_negative = sum(1 for gold, pred in zip(labels, predictions) if gold == label and pred != label)
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
        scores.append((2 * precision * recall / (precision + recall)) if precision + recall else 0.0)
    return sum(scores) / len(scores)


def expected_calibration_error(labels: list[str], predictions: list[str | None], confidences: list[float], bins: int = 10) -> float:
    if not labels:
        return 0.0
    total = len(labels)
    ece = 0.0
    for bucket in range(bins):
        lower = bucket / bins
        upper = (bucket + 1) / bins
        indices = [
            index for index, confidence in enumerate(confidences)
            if lower <= confidence < upper or (bucket == bins - 1 and confidence == 1.0)
        ]
        if not indices:
            continue
        bucket_accuracy = sum(1 for index in indices if labels[index] == predictions[index]) / len(indices)
        bucket_confidence = sum(confidences[index] for index in indices) / len(indices)
        ece += len(indices) / total * abs(bucket_accuracy - bucket_confidence)
    return ece


def brier_score(labels: list[str], predictions: list[str | None], confidences: list[float]) -> float:
    if not labels:
        return 0.0
    total = 0.0
    for label, prediction, confidence in zip(labels, predictions, confidences):
        confidence = clamp_confidence(confidence)
        for candidate in LABELS:
            target = 1.0 if candidate == label else 0.0
            if prediction == candidate:
                probability = confidence
            else:
                probability = (1.0 - confidence) / (len(LABELS) - 1)
            total += (probability - target) ** 2
    return total / len(labels)


def clamp_confidence(value: float) -> float:
    if not isfinite(value):
        return 0.0
    return max(0.0, min(1.0, float(value)))


def confusion_matrix(labels: list[str], predictions: list[str | None]) -> dict[str, dict[str, int]]:
    matrix = {label: {candidate: 0 for candidate in LABELS} for label in LABELS}
    for label, prediction in zip(labels, predictions):
        if prediction in LABELS:
            matrix[label][prediction] += 1
    return matrix


def evaluate_predictions(rows: Iterable[dict], baseline_rows: Iterable[dict] | None = None) -> dict:
    materialized = list(rows)
    labels = [str(row["label"]).lower() for row in materialized]
    predictions = [parse_prediction(str(row.get("prediction", ""))) for row in materialized]
    confidences = [clamp_confidence(float(row.get("confidence", 0.0))) for row in materialized]
    latencies = [float(row.get("latency_ms", 0.0)) for row in materialized]

    invalid_count = sum(1 for prediction in predictions if prediction is None)
    unsupported_tokens = ("buy", "sell", "upgrade", "downgrade", "price target")
    hallucination_count = sum(
        1 for row, prediction in zip(materialized, predictions)
        if prediction is None or any(token in str(row.get("prediction", "")).lower() for token in unsupported_tokens)
    )

    metrics = {
        "examples": len(materialized),
        "accuracy": round(accuracy_score(labels, predictions), 4),
        "macro_f1": round(macro_f1_score(labels, predictions), 4),
        "expected_calibration_error": round(expected_calibration_error(labels, predictions, confidences), 4),
        "brier_score": round(brier_score(labels, predictions, confidences), 4),
        "invalid_label_rate": round(invalid_count / len(materialized), 4) if materialized else 0.0,
        "hallucination_proxy_rate": round(hallucination_count / len(materialized), 4) if materialized else 0.0,
        "mean_latency_ms": round(sum(latencies) / len(latencies), 2) if latencies else 0.0,
        "confusion_matrix": confusion_matrix(labels, predictions),
        "label_distribution": dict(Counter(labels)),
    }

    if baseline_rows is not None:
        baseline_metrics = evaluate_predictions(list(baseline_rows))
        baseline_accuracy = baseline_metrics["accuracy"]
        lift = (metrics["accuracy"] - baseline_accuracy) / baseline_accuracy if baseline_accuracy else 0.0
        metrics["baseline_accuracy"] = baseline_accuracy
        metrics["relative_accuracy_lift"] = round(lift, 4)
    return metrics
