# Tech Challenge IADT — Fase 3 Summary

> Source: `Tech_Challenge_IADT_Fase3.pdf` (5 pages)
> Extracted: 2026-03-22

---

## Desafio

Criar um **assistente virtual médico** treinado com dados do hospital, capaz de:
- Auxiliar nas condutas clínicas
- Responder dúvidas de médicos
- Sugerir procedimentos com base em protocolos internos
- Coordenar fluxos de decisão automatizados com LangChain (ex: verificar exames, sugerir tratamentos, emitir alertas)

---

## Requisitos Obrigatórios

### 1. Fine-tuning de LLM com dados médicos internos
- Modelo base: LLaMA, Falcon ou similar
- Dados de treinamento:
  - Protocolos médicos do hospital
  - Perguntas frequentes de médicos
  - Modelos de laudos, receitas e procedimentos internos
- Técnicas obrigatórias: preprocessing, anonimização e curadoria dos dados

### 2. Assistente médico com LangChain
- Pipeline integrando a **LLM customizada** (fine-tuned)
- Consultas em base de dados estruturadas (prontuários, registros)
- Respostas contextualizadas com informações atualizadas do paciente
- Fluxos com **LangGraph**

### 3. Segurança e validação
- Limites de atuação: **nunca prescrever diretamente sem validação humana**
- Logging detalhado para rastreamento e auditoria
- **Explainability**: indicar a fonte da informação usada na resposta

### 4. Organização do código
- Projeto modularizado em Python
- README com instruções completas

---

## Entregáveis

### Repositório Git
- [ ] Pipeline de fine-tuning
- [ ] Integração com LangChain
- [ ] Fluxos do LangGraph
- [ ] Dataset anonimizado ou dados sintéticos de exemplo

### Relatório Técnico (Markdown)
- [ ] Explicação do processo de fine-tuning
- [ ] Descrição do assistente médico criado
- [ ] **Diagrama do fluxo LangChain/LangGraph**
- [ ] Avaliação do modelo e análise dos resultados

### Vídeo (até 15 minutos)
- [ ] Treinamento e funcionamento da LLM personalizada
- [ ] Execução de um fluxo automatizado
- [ ] Resposta a perguntas clínicas contextualizadas
- [ ] Logs e validação das respostas

---

## Datasets Sugeridos

| Dataset   | Conteúdo                                      | URL                                   |
|-----------|-----------------------------------------------|---------------------------------------|
| PubMedQA  | Perguntas e respostas clínicas (publicações)  | https://pubmedqa.github.io/           |
| MedQuAD   | Perguntas e respostas sobre saúde             | https://github.com/abachaa/MedQuAD    |

---

## Mapeamento para Milestones (`fase3_milestones.md`)

| Requisito do PDF                          | Milestone |
|-------------------------------------------|-----------|
| Fine-tuning + preprocessing/anonimização  | M1 + M2   |
| Pipeline LangChain com LLM customizada    | M3        |
| LangGraph fluxos automatizados            | M3        |
| Segurança, limites, logging, explainability | M4      |
| README + Relatório + Vídeo                | M5        |

---

## Decisões Chave de Design

### LLM na geração de respostas
O PDF exige **integrar a LLM customizada** (fine-tuned) no pipeline.
Recomendação de arquitetura:
- **TinyLlama fine-tuned** (M2): usado como gerador principal de resposta médica
- **Gemini**: usado para explicabilidade (`explainer.py`) e fallback quando o adapter não está disponível
- Isso atende ao requisito de "integrar a LLM customizada" sem sacrificar qualidade

### Fluxo LangGraph obrigatório
O PDF cita explicitamente LangGraph. Os 4 nós do milestone são suficientes:
```
classify_intent → retrieve_context → generate_response → validate_response
```

### Explainability (requisito explícito)
Toda resposta deve citar a **fonte** usada (documento do KB recuperado).
Implementado em `src/assistant/chain.py` via `source_documents` + `src/assistant/explainer.py`.

### Limite de prescrição (requisito explícito)
Guard-rail obrigatório: assistente **nunca prescreve diretamente**.
Disclaimer padrão obrigatório em toda resposta.
Implementado em `src/assistant/guardrails.py` (M4).

---

## Nota sobre Peso da Atividade

> "Esta atividade é obrigatória, valendo **90% da nota** de todas as disciplinas da fase."
