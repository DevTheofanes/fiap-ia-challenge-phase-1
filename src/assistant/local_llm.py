"""Local assistant model wrapper for TinyLlama + LoRA adapter."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.config import FINETUNE_FINAL_ADAPTER_DIR
from src.llm.finetune_config import FinetuneConfig


class LocalModelInitializationError(RuntimeError):
    """Raised when the local assistant model cannot be initialized."""


class LocalModelOutputError(RuntimeError):
    """Raised when the local model generates unusable output (garbage/repetition)."""


def _is_garbage(text: str) -> bool:
    """Return True if the output is clearly structured garbage (key=value spam).

    Only blocks outputs with a high ratio of `=` signs, which indicates the model
    generated structured data instead of natural language. Medical text that repeats
    clinical terms is NOT considered garbage.
    """
    if not text:
        return True

    # key=value spam (e.g. "tumor_stage=IIIA; tumor_stage=IIIA; ...")
    eq_ratio = text.count("=") / max(len(text), 1)
    return eq_ratio > 0.08


@dataclass
class LocalAssistantLLM:
    """Thin inference wrapper around the fine-tuned TinyLlama adapter."""

    adapter_dir: Path = FINETUNE_FINAL_ADAPTER_DIR
    config: FinetuneConfig = field(default_factory=FinetuneConfig)
    _tokenizer: Any | None = None
    _model: Any | None = None

    def _load(self) -> None:
        if self._model is not None and self._tokenizer is not None:
            return

        if not self.adapter_dir.exists():
            raise LocalModelInitializationError(
                f"Adapter not found at {self.adapter_dir}. Run scripts/fine_tune.py first."
            )

        try:
            from peft import PeftModel
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except Exception as exc:  # pragma: no cover - dependency-level failure
            raise LocalModelInitializationError(
                "Local model dependencies are unavailable. Install transformers and peft."
            ) from exc

        try:
            tokenizer = AutoTokenizer.from_pretrained(self.config.base_model_name)
            tokenizer.pad_token = tokenizer.eos_token

            base_model = AutoModelForCausalLM.from_pretrained(
                self.config.base_model_name,
                device_map="auto",
            )
            model = PeftModel.from_pretrained(base_model, str(self.adapter_dir))
            model.eval()
        except Exception as exc:  # pragma: no cover - hardware/model-level failure
            raise LocalModelInitializationError(
                f"Unable to initialize local assistant model: {exc}"
            ) from exc

        self._tokenizer = tokenizer
        self._model = model

    def generate(self, prompt: str, max_new_tokens: int = 256) -> str:
        """Generate an assistant answer from the local fine-tuned model."""
        self._load()
        assert self._tokenizer is not None
        assert self._model is not None

        try:
            import torch
        except Exception as exc:  # pragma: no cover - dependency-level failure
            raise RuntimeError("torch is required for local assistant inference.") from exc

        inputs = self._tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048,  # TinyLlama's full context window; max_seq_length=256 is training-only
        )
        inputs = {key: value.to(self._model.device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.3,
                top_p=0.9,
                pad_token_id=self._tokenizer.eos_token_id,
            )

        generated = outputs[0][inputs["input_ids"].shape[1]:]
        text = self._tokenizer.decode(generated, skip_special_tokens=True).strip()

        # TinyLlama sometimes echoes prompt sections back; truncate at the first
        # occurrence of any section header that belongs to the input prompt.
        _STOP_PHRASES = [
            "Patient Context",
            "Knowledge Base Context",
            "Question\n",
            "\n\nAnswer:",
        ]
        for phrase in _STOP_PHRASES:
            idx = text.find(phrase)
            if idx > 0:
                text = text[:idx].strip()

        if _is_garbage(text):
            raise LocalModelOutputError(
                "Local model produced unusable output — triggering fallback."
            )

        return text


def allow_fallback() -> bool:
    return os.getenv("ASSISTANT_ALLOW_FALLBACK", "false").lower() == "true"
