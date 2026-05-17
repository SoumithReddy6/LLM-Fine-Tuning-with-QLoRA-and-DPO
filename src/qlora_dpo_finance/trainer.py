"""Training utility functions for QLoRA and DPO scripts."""

from __future__ import annotations

from typing import Any


def build_quantization_config(qlora_config: dict[str, Any]):
    try:
        import torch
        from transformers import BitsAndBytesConfig
    except Exception as exc:
        raise RuntimeError("Install torch, transformers, and bitsandbytes for QLoRA training") from exc

    dtype_name = str(qlora_config.get("bnb_4bit_compute_dtype", "bfloat16"))
    compute_dtype = torch.bfloat16 if dtype_name == "bfloat16" else torch.float16
    return BitsAndBytesConfig(
        load_in_4bit=bool(qlora_config.get("load_in_4bit", True)),
        bnb_4bit_quant_type=str(qlora_config.get("bnb_4bit_quant_type", "nf4")),
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=bool(qlora_config.get("bnb_4bit_use_double_quant", True)),
    )


def build_lora_config(lora_config: dict[str, Any]):
    try:
        from peft import LoraConfig
    except Exception as exc:
        raise RuntimeError("Install peft for LoRA training") from exc

    return LoraConfig(
        r=int(lora_config.get("r", 16)),
        lora_alpha=int(lora_config.get("alpha", 32)),
        lora_dropout=float(lora_config.get("dropout", 0.05)),
        target_modules=list(lora_config.get("target_modules", [])) or None,
        bias="none",
        task_type="CAUSAL_LM",
    )


def load_model_and_tokenizer(config):
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as exc:
        raise RuntimeError("Install transformers to load models") from exc

    model_config = config.section("model")
    tokenizer = AutoTokenizer.from_pretrained(
        model_config["name_or_path"],
        trust_remote_code=bool(model_config.get("trust_remote_code", False)),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_config["name_or_path"],
        quantization_config=build_quantization_config(config.section("qlora")),
        device_map="auto",
        trust_remote_code=bool(model_config.get("trust_remote_code", False)),
    )
    model.config.use_cache = False
    return model, tokenizer
