"""FastAPI inference service for financial-sentiment classification.

Serving lifecycle stage: exposes the fine-tuned model behind a REST endpoint with
health check, latency, and a batch route. Backend selected by MODEL_BACKEND
(`ollama` for a free local demo, `transformers` to serve the trained LoRA adapter).

Run:  uvicorn qlora_dpo_finance.api:app --reload
"""

from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel, Field

from qlora_dpo_finance.serving import SentimentService


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1)


class BatchRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1)


app = FastAPI(
    title="Financial Sentiment Inference",
    version="0.1.0",
    description="Serves the QLoRA/DPO fine-tuned financial-sentiment model (transformers adapter or local Ollama backend).",
)
_service: SentimentService | None = None


def get_service() -> SentimentService:
    global _service
    if _service is None:
        _service = SentimentService()
    return _service


@app.get("/health")
def health() -> dict:
    svc = get_service()
    return {"status": "ok", "backend": svc.backend, "model": svc.model}


@app.post("/predict")
def predict(payload: PredictRequest) -> dict:
    pred = get_service().predict(payload.text)
    return pred.__dict__


@app.post("/predict/batch")
def predict_batch(payload: BatchRequest) -> dict:
    svc = get_service()
    preds = [svc.predict(t).__dict__ for t in payload.texts]
    return {"count": len(preds), "predictions": preds}
