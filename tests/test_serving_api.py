"""Serving API tests with a mock backend (no Ollama/model needed in CI)."""

from __future__ import annotations

from fastapi.testclient import TestClient

from qlora_dpo_finance import api
from qlora_dpo_finance.serving import Prediction


class FakeService:
    backend = "fake"
    model = "fake-model"

    def predict(self, text: str) -> Prediction:
        label = "positive" if "profit" in text.lower() else "negative"
        return Prediction(label=label, raw=label, confidence=0.9, latency_ms=1.0, backend="fake", model="fake-model")


def _client() -> TestClient:
    api._service = FakeService()
    return TestClient(api.app)


def test_health_reports_backend():
    resp = _client().get("/health")
    assert resp.status_code == 200
    assert resp.json()["backend"] == "fake"


def test_predict_returns_label_and_latency():
    resp = _client().post("/predict", json={"text": "Record profit and raised guidance"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["label"] == "positive"
    assert "latency_ms" in body


def test_batch_predict():
    resp = _client().post("/predict/batch", json={"texts": ["profit up", "losses mount"]})
    assert resp.status_code == 200
    assert resp.json()["count"] == 2
