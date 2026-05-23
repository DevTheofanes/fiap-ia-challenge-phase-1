from __future__ import annotations

import json
import os
import re
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
    FALLBACK_MODELS = ("gemini-2.0-flash", "gemini-flash-latest")

    def __init__(self, api_key: str, model: str) -> None:
        self.api_key = api_key
        self.model = model
        self._client = None

    def _init_client(self, model: str | None = None) -> Any:
        try:
            import google.generativeai as genai  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "google-generativeai is not installed. Install it to use Gemini."
            ) from exc
        selected_model = model or self.model
        genai.configure(api_key=self.api_key)
        self._client = genai.GenerativeModel(selected_model)
        self.model = selected_model
        return self._client

    def generate(self, prompt: str, *, temperature: float = 0.2, max_tokens: int = 512) -> LLMResponse:
        try:
            import google.generativeai as genai  # type: ignore
        except Exception:
            genai = None

        def _run_generate(client: Any) -> Any:
            if genai and hasattr(genai, "types"):
                config = genai.types.GenerationConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                )
                return client.generate_content(prompt, generation_config=config)
            return client.generate_content(prompt)

        client = self._client or self._init_client()
        try:
            response = _run_generate(client)
        except Exception as exc:
            message = str(exc)
            if "404" not in message:
                raise
            for fallback_model in self.FALLBACK_MODELS:
                if fallback_model == self.model:
                    continue
                client = self._init_client(fallback_model)
                try:
                    response = _run_generate(client)
                    break
                except Exception as retry_exc:
                    if "404" not in str(retry_exc):
                        raise
            else:
                raise
        text = getattr(response, "text", None) or str(response)
        return LLMResponse(text=text, raw=response, model=self.model, provider="gemini")


class OpenAIClient(LLMClient):
    def __init__(self, api_key: str, model: str) -> None:
        self.api_key = api_key
        self.model = model
        self._client = None

    def _init_client(self) -> Any:
        try:
            from openai import OpenAI  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("openai is not installed. Install it to use OpenAI.") from exc
        self._client = OpenAI(api_key=self.api_key)
        return self._client

    def generate(self, prompt: str, *, temperature: float = 0.2, max_tokens: int = 512) -> LLMResponse:
        client = self._client or self._init_client()
        response = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        text = response.choices[0].message.content or ""
        return LLMResponse(text=text, raw=response, model=self.model, provider="openai")


class MockClient(LLMClient):
    _call_counter: int = 0

    def generate(self, prompt: str, *, temperature: float = 0.2, max_tokens: int = 512) -> LLMResponse:
        # Return a JSON array when the prompt requests synthetic Q&A pairs
        if "JSON array" in prompt and "question" in prompt and "answer" in prompt:
            # Extract requested count from prompt; default to 10
            m = re.search(r"exactly (\d+) pairs", prompt)
            n = int(m.group(1)) if m else 10
            offset = MockClient._call_counter * n
            MockClient._call_counter += 1
            pairs = [
                {
                    "question": f"Mock oncology question {offset + i}: What is the standard treatment for breast cancer variant {offset + i}?",
                    "answer": f"Mock answer {offset + i}: Treatment options include surgery, radiation therapy, and adjuvant systemic therapy based on tumor characteristics and stage.",
                }
                for i in range(n)
            ]
            return LLMResponse(text=json.dumps(pairs), raw=None, model="mock", provider="mock")
        payload = {
            "summary_for_clinician": [
                "Mock response: LLM disabled.",
                "Set OPENAI_API_KEY or GEMINI_API_KEY to enable a real LLM.",
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
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if openai_api_key:
        openai_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        return OpenAIClient(api_key=openai_api_key, model=openai_model)
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if gemini_api_key:
        gemini_model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
        return GeminiClient(api_key=gemini_api_key, model=gemini_model)
    return None
