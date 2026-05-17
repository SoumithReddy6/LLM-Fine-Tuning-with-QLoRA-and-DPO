"""Experiment tracking helpers."""

from __future__ import annotations

from typing import Any


def log_to_mlflow(experiment_name: str, params: dict[str, Any], metrics: dict[str, Any]) -> None:
    try:
        import mlflow
    except Exception:
        return
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run():
        mlflow.log_params(flatten(params))
        scalar_metrics = {key: value for key, value in metrics.items() if isinstance(value, (int, float))}
        mlflow.log_metrics(scalar_metrics)


def log_to_wandb(project: str, config: dict[str, Any], metrics: dict[str, Any]) -> None:
    try:
        import wandb
    except Exception:
        return
    run = wandb.init(project=project, config=config)
    run.log(metrics)
    run.finish()


def flatten(payload: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    items: dict[str, Any] = {}
    for key, value in payload.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            items.update(flatten(value, name))
        elif isinstance(value, (str, int, float, bool)):
            items[name] = value
    return items
