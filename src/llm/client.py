from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ENV_LOADED = False


def _load_env_file(env_path: Path | None = None) -> None:
    global ENV_LOADED
    if ENV_LOADED:
        return
    env_file = env_path or Path.cwd() / ".env"
    if not env_file.exists():
        ENV_LOADED = True
        return
    try:
        for line in env_file.read_text().splitlines():
            raw = line.strip()
            if not raw or raw.startswith("#") or "=" not in raw:
                continue
            key, value = raw.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"')
            os.environ.setdefault(key, value)
    finally:
        ENV_LOADED = True


@dataclass
class LLMResponse:
    text: str
    raw: Any | None = None
    model: str | None = None
    provider: str | None = None


class LLMClient:
    def generate(self, prompt: str, *, temperature: float = 0.2, max_tokens: int = 512) -> LLMResponse:
        raise NotImplementedError


class GeminiClient(LLMClient):
    def __init__(self, api_key: str, model: str) -> None:
        self.api_key = api_key
        self.model = model
        self._client = None

    def _init_client(self) -> Any:
        try:
            import google.generativeai as genai  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "google-generativeai is not installed. Install it to use Gemini."
            ) from exc
        genai.configure(api_key=self.api_key)
        self._client = genai.GenerativeModel(self.model)
        return self._client

    def generate(self, prompt: str, *, temperature: float = 0.2, max_tokens: int = 512) -> LLMResponse:
        client = self._client or self._init_client()
        try:
            import google.generativeai as genai  # type: ignore
        except Exception:
            genai = None
        if genai and hasattr(genai, "types"):
            config = genai.types.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
            )
            response = client.generate_content(prompt, generation_config=config)
        else:
            response = client.generate_content(prompt)
        text = getattr(response, "text", None) or str(response)
        return LLMResponse(text=text, raw=response, model=self.model, provider="gemini")


class MockClient(LLMClient):
    def generate(self, prompt: str, *, temperature: float = 0.2, max_tokens: int = 512) -> LLMResponse:
        payload = {
            "summary_for_clinician": [
                "Mock response: LLM disabled.",
                "Check GEMINI_API_KEY to enable Gemini.",
            ],
            "key_factors": ["mock_feature"],
            "recommended_next_steps": ["review case manually"],
            "limitations_and_caveats": ["mock output"],
            "confidence_statement": "mock response",
        }
        return LLMResponse(text=json.dumps(payload), raw=None, model="mock", provider="mock")


def get_llm_client() -> LLMClient | None:
    _load_env_file()
    use_mock = os.getenv("LLM_USE_MOCK", "false").lower() == "true"
    if use_mock:
        return MockClient()
    api_key = os.getenv("GEMINI_API_KEY")
    model = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
    if not api_key:
        return None
    return GeminiClient(api_key=api_key, model=model)
