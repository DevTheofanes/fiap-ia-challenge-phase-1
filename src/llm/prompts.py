from __future__ import annotations

import json
from typing import Any

CASE_PROMPT_TEMPLATE = """
Voce e um assistente clinico que explica resultados de modelos de ML para medicos de triagem.
Regras:
- Linguagem objetiva, sem alarmismo.
- Nao inventar fatos.
- Explicitar incerteza.
- Nao prescrever tratamento.
- Sempre incluir limitacoes e aviso de que nao substitui julgamento clinico.

Responda SOMENTE em JSON válido com as chaves abaixo:
- summary_for_clinician: lista de 3 a 6 frases curtas
- key_factors: lista curta
- recommended_next_steps: lista curta, não prescritiva
- limitations_and_caveats: lista curta
- confidence_statement: string curta

Entrada (JSON):
{payload}
""".strip()

METRICS_PROMPT_TEMPLATE = """
Voce e um analista de ML e precisa resumir resultados comparativos.
Regras:
- Linguagem direta e tecnica.
- Descrever ganhos e trade-offs.
- Nao inventar metricas.
- Mencionar limitacoes e incerteza.

Responda SOMENTE em JSON válido com as chaves:
- summary
- key_improvements
- tradeoffs
- limitations

Tabela de metricas:
{table}
""".strip()


def format_case_prompt(payload: dict[str, Any]) -> str:
    return CASE_PROMPT_TEMPLATE.format(payload=json.dumps(payload, ensure_ascii=False, indent=2))


def format_metrics_prompt(table_markdown: str) -> str:
    return METRICS_PROMPT_TEMPLATE.format(table=table_markdown)


SYNTHETIC_QA_PROMPT_TEMPLATE = """
You are a medical education assistant. Generate exactly {n} oncology question-and-answer pairs
covering topics such as: breast cancer diagnosis, tumor staging, biopsy interpretation,
cancer biomarkers, treatment modalities, prognosis factors, screening guidelines, and pathology.

Rules:
- Each pair must be clinically realistic and educationally valuable.
- Do NOT include any real patient names, dates, hospital names, or identifiable information.
- Use generic placeholders like "the patient" or "a 55-year-old female".
- Answers should be factual, evidence-based, and 2-6 sentences long.
- Respond ONLY with a valid JSON array. No markdown, no preamble, no trailing text.

Format:
[
  {{"question": "...", "answer": "..."}},
  ...
]

Generate exactly {n} pairs now.
""".strip()


def format_synthetic_qa_prompt(n: int) -> str:
    return SYNTHETIC_QA_PROMPT_TEMPLATE.format(n=n)
