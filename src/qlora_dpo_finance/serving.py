"""Inference serving for the fine-tuned financial-sentiment model.

Two backends behind one interface (the serving lifecycle stage):

- ``transformers``: loads the base model + the trained LoRA adapter via PEFT and
  serves the actual fine-tuned model. This is the production path. (Requires
  ``transformers`` + ``peft``; point ``ADAPTER_PATH`` at the adapter downloaded
  from the training run.)
- ``ollama``: calls a local Ollama model (free, no GPU, no API key) for a
  runnable demo on machines without the adapter/CUDA. Serves the base model with
  the same sentiment prompt, so the API contract is identical.

Both return a parsed label, a confidence, and latency — the contract a downstream
service consumes regardless of backend.
"""

from __future__ import annotations

import json
import os
import time
import urllib.request
from dataclasses import dataclass

from qlora_dpo_finance.data import format_prompt
from qlora_dpo_finance.metrics import parse_prediction


@dataclass
class Prediction:
    label: str | None
    raw: str
    confidence: float
    latency_ms: float
    backend: str
    model: str


class SentimentService:
    def __init__(self, backend: str | None = None) -> None:
        self.backend = (backend or os.getenv("MODEL_BACKEND", "ollama")).lower()
        self._tf = None  # lazy transformers pipeline
        if self.backend == "transformers":
            self._init_transformers()
        else:
            self.ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
            self.model = os.getenv("OLLAMA_MODEL", "qwen2.5:0.5b")

    # --- transformers backend (serves the real fine-tuned adapter) ---
    def _init_transformers(self) -> None:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        base = os.getenv("BASE_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
        adapter = os.getenv("ADAPTER_PATH", "outputs/qwen25_0_5b_qlora_rank16")
        self.model = f"{base}+{adapter}"
        self._tokenizer = AutoTokenizer.from_pretrained(base)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.float32, device_map="cpu")
        if os.path.isdir(adapter):
            model = PeftModel.from_pretrained(model, adapter)
        model.eval()
        self._tf = model

    def _predict_transformers(self, text: str) -> tuple[str, float]:
        import torch

        prompt = format_prompt(text)
        inputs = self._tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            out = self._tf.generate(
                **inputs, max_new_tokens=5, do_sample=False,
                return_dict_in_generate=True, output_scores=True,
                pad_token_id=self._tokenizer.pad_token_id,
            )
        new_ids = out.sequences[0][inputs["input_ids"].shape[1]:]
        completion = self._tokenizer.decode(new_ids, skip_special_tokens=True).strip()
        probs = [torch.softmax(s[0], dim=-1)[t].item() for t, s in zip(new_ids, out.scores)]
        confidence = round(sum(probs) / len(probs), 4) if probs else 0.0
        return completion, confidence

    # --- ollama backend (free local demo) ---
    def _predict_ollama(self, text: str) -> tuple[str, float]:
        payload = {
            "model": self.model,
            "prompt": format_prompt(text),
            "system": "You are a financial-sentiment classifier. Reply with exactly one word: negative, neutral, or positive.",
            "stream": False,
            "options": {"temperature": 0.0, "num_predict": 5},
        }
        req = urllib.request.Request(
            f"{self.ollama_host}/api/generate",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            body = json.load(resp)
        return body.get("response", "").strip(), 1.0  # ollama generate has no per-token prob here

    def predict(self, text: str) -> Prediction:
        started = time.perf_counter()
        if self.backend == "transformers":
            raw, confidence = self._predict_transformers(text)
        else:
            raw, confidence = self._predict_ollama(text)
        latency_ms = round((time.perf_counter() - started) * 1000, 2)
        return Prediction(
            label=parse_prediction(raw),
            raw=raw,
            confidence=confidence,
            latency_ms=latency_ms,
            backend=self.backend,
            model=self.model,
        )
