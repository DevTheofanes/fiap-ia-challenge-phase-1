from __future__ import annotations

import os
from pathlib import Path

from src.llm.client import LLMClient, get_llm_client


CLINICAL_AUDIO_PROMPT = """\
Voce e um assistente clinico para triagem de saude materna no pos-parto.
Analise a transcricao abaixo e produza um laudo estruturado em portugues.

Cubra obrigatoriamente:
- sinais compativeis com depressao pos-parto;
- sinais de ansiedade;
- possiveis sinais de violencia, coercao, negligencia ou falta de seguranca;
- fadiga hormonal, privacao de sono e exaustao fisica/emocional.

Formato esperado:
1. Resumo clinico
2. Sinais observados por categoria
3. Nivel de alerta: baixo, moderado ou alto
4. Perguntas de acompanhamento recomendadas
5. Encaminhamentos sugeridos
6. Limitacoes

Regras:
- Nao faca diagnostico definitivo.
- Diferencie evidencias explicitas de hipoteses.
- Se houver risco de autoagressao, violencia ou inseguranca imediata, destaque a necessidade de suporte profissional urgente.
- Seja objetivo, cuidadoso e orientado a decisao clinica.

Transcricao:
\"\"\"{transcript}\"\"\"
"""


def transcribe(audio_path: str | Path, *, model: str = "whisper-1") -> str:
    """Transcribe an audio file through OpenAI's transcription API."""
    path = Path(audio_path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")
    if not path.is_file():
        raise ValueError(f"Audio path is not a file: {path}")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required to transcribe audio with OpenAI.")

    try:
        from openai import OpenAI  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("openai is not installed. Install it to transcribe audio.") from exc

    client = OpenAI(api_key=api_key)
    with path.open("rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            file=audio_file,
            model=model,
            response_format="text",
        )
    return str(transcript).strip()


def analyze_transcript(transcript: str, client: LLMClient | None = None) -> str:
    """Generate a structured clinical report from a transcript."""
    cleaned_transcript = transcript.strip()
    if not cleaned_transcript:
        raise ValueError("Transcript cannot be empty.")

    llm_client = client or get_llm_client()
    if llm_client is None:
        raise RuntimeError(
            "No LLM client configured. Set OPENAI_API_KEY, GEMINI_API_KEY, or LLM_USE_MOCK=true."
        )

    prompt = CLINICAL_AUDIO_PROMPT.format(transcript=cleaned_transcript)
    response = llm_client.generate(prompt, temperature=0.1, max_tokens=1500)
    return response.text.strip()
