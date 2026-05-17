"""Configuration loading helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ExperimentConfig:
    path: Path
    raw: dict[str, Any]

    @property
    def name(self) -> str:
        return str(self.raw.get("experiment_name", self.path.stem))

    @property
    def stage(self) -> str:
        return str(self.raw.get("stage", "eval"))

    @property
    def task(self) -> str:
        return str(self.raw.get("task", "financial_sentiment"))

    def section(self, name: str) -> dict[str, Any]:
        value = self.raw.get(name, {})
        if not isinstance(value, dict):
            raise ValueError(f"Config section '{name}' must be a mapping")
        return value


def load_config(path: str | Path) -> ExperimentConfig:
    config_path = Path(path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Config {config_path} must contain a YAML mapping")
    return ExperimentConfig(path=config_path, raw=payload)
