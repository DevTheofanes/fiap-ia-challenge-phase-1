# Fase 3 — Milestones de Implementação
## FIAP IADT Tech Challenge — Medical AI Assistant

> **Contexto:** Este documento descreve o plano de implementação da Fase 3 do desafio, que estende o pipeline de diagnóstico de câncer de mama (Fases 1 e 2) com fine-tuning de LLM, assistente médico com LangChain/LangGraph, segurança, rastreabilidade e relatório técnico.

---

## Visão Geral da Fase 3

A Fase 3 tem quatro entregas obrigatórias:

1. **Fine-tuning de LLM** com dados médicos sintéticos/públicos
2. **Pipeline de assistente médico** com LangChain e fluxos LangGraph
3. **Segurança, explicabilidade e auditoria** (guard-rails, citações, logs)
4. **Relatório técnico + README atualizado + vídeo demo**

O código deve estar em um repositório Git público com commits organizados por etapa.

---

## Milestones

### M1 — Preparação de Dados para Fine-Tuning

**Objetivo:** Produzir um dataset curado em JSONL pronto para treinamento supervisionado (SFT).

**Tarefas:**

- [ ] Identificar e baixar dataset público (ex.: [MedQuAD](https://github.com/abachaa/MedQuAD), [PubMedQA](https://pubmedqa.github.io/), ou geração sintética com Gemini)
- [ ] Implementar `scripts/prepare_finetune_data.py`:
  - Filtragem de perguntas/respostas relacionadas a oncologia e diagnóstico
  - Anonimização (remoção de nomes, datas, IDs) via regex ou `presidio`
  - Conversão para formato de instrução: `{"prompt": "...", "completion": "..."}`
- [ ] Salvar dataset tratado em `data/finetune/train.jsonl` e `data/finetune/val.jsonl`
- [ ] Documentar proveniência e licença dos dados em `data/finetune/README.md`

**Arquivos envolvidos:**
```
data/finetune/
  train.jsonl
  val.jsonl
  README.md
scripts/prepare_finetune_data.py
```

**Ferramentas sugeridas:** `datasets` (HuggingFace), `presidio-analyzer`, `pandas`

**Critério de conclusão (Done when):**
- `data/finetune/train.jsonl` contém ≥ 500 pares instrução/resposta médicos
- Script roda sem erros e gera split treino/validação reproduzível com `RANDOM_STATE=42`

---

### M2 — Pipeline de Fine-Tuning do LLM

**Objetivo:** Treinar (ou adaptar via LoRA/QLoRA) um LLM base com os dados médicos preparados no M1.

**Tarefas:**

- [ ] Escolher modelo base (ex.: `google/flan-t5-base`, `mistralai/Mistral-7B-v0.1`, ou `BioMistral-7B`)
- [ ] Implementar `scripts/fine_tune.py` usando `transformers` + `peft` + `trl`:
  - Configuração de `SFTTrainer` com LoRA (`r=8`, `lora_alpha=16`)
  - Logging de métricas de treinamento para `artifacts/logs/finetune.jsonl` (padrão `src/logging_utils.py`)
  - Checkpoint automático a cada época em `artifacts/finetune_checkpoints/`
- [ ] Avaliar modelo fine-tunado vs. base em amostra do val set (BLEU, ROUGE ou avaliação manual)
- [ ] Salvar adapter weights em `artifacts/finetune_checkpoints/final_adapter/`

**Arquivos envolvidos:**
```
scripts/fine_tune.py
scripts/eval_finetune.py
artifacts/finetune_checkpoints/
artifacts/logs/finetune.jsonl
src/llm/finetune_config.py   # HyperParams: batch_size, lr, epochs, lora_r, etc.
```

**Ferramentas sugeridas:** `transformers`, `peft`, `trl`, `bitsandbytes` (QLoRA), `accelerate`

**Critério de conclusão (Done when):**
- `scripts/fine_tune.py` executa end-to-end sem erro
- Checkpoint final salvo e carregável com `PeftModel.from_pretrained()`
- Log de treino registrado em JSONL seguindo padrão existente de `src/logging_utils.py`

---

### M3 — Assistente Médico com LangChain + LangGraph

**Objetivo:** Construir pipeline de assistente conversacional que usa o modelo fine-tunado + recuperação de contexto (RAG) com fluxo de decisão via LangGraph.

**Tarefas:**

- [ ] Criar módulo `src/assistant/`:
  - `src/assistant/retriever.py` — vector store com documentos médicos (ChromaDB ou FAISS)
  - `src/assistant/chain.py` — LangChain `RetrievalQA` ou LCEL chain com prompt template
  - `src/assistant/graph.py` — LangGraph `StateGraph` com nós: `classify_intent → retrieve_context → generate_response → validate_response`
  - `src/assistant/tools.py` — tools opcionais (busca PubMed, lookup de CID-10)
- [ ] Implementar `scripts/run_assistant.py` — CLI interativo para testar o assistente
- [ ] Indexar documentos de referência em `data/kb/` (knowledge base) no vector store
- [ ] Integrar predições do modelo ML das Fases 1/2 como contexto adicional para o LLM

**Arquivos envolvidos:**
```
src/assistant/
  __init__.py
  retriever.py
  chain.py
  graph.py
  tools.py
scripts/run_assistant.py
data/kb/                     # Documentos médicos de referência (PDFs, TXTs)
artifacts/vectorstore/       # Índice persistido do ChromaDB/FAISS
```

**Ferramentas sugeridas:** `langchain`, `langgraph`, `langchain-community`, `chromadb` ou `faiss-cpu`, `langchain-huggingface`

**Critério de conclusão (Done when):**
- `python scripts/run_assistant.py` responde perguntas médicas com contexto recuperado do KB
- LangGraph executa os 4 nós do fluxo (classify → retrieve → generate → validate)
- Resposta inclui citação da fonte recuperada

---

### M4 — Segurança, Explicabilidade e Auditoria

**Objetivo:** Adicionar guard-rails de segurança, citações de fontes nas respostas e logs de auditoria rastreáveis.

**Tarefas:**

- [ ] Implementar `src/assistant/guardrails.py`:
  - Filtro de entrada: bloquear perguntas fora do escopo médico ou com conteúdo inapropriado
  - Filtro de saída: detectar e suprimir diagnósticos definitivos sem disclaimer
  - Adicionar disclaimer padrão em toda resposta: *"Esta informação é educacional e não substitui consulta médica."*
- [ ] Adicionar citações de fonte (`source_documents`) na resposta final do LangChain chain
- [ ] Implementar `src/assistant/audit_logger.py`:
  - Registrar em `artifacts/logs/audit.jsonl`: timestamp, `user_query`, `retrieved_docs`, `model_response`, `guardrail_triggered` (bool)
  - Seguir o padrão de `src/logging_utils.py`
- [ ] Implementar `src/assistant/explainer.py`:
  - Wrapper que chama Gemini (existente em `src/llm/`) para explicar a resposta do modelo fine-tunado
  - Conectar com predições do pipeline ML (probabilidade de malignidade, feature importances do RF)

**Arquivos envolvidos:**
```
src/assistant/guardrails.py
src/assistant/audit_logger.py
src/assistant/explainer.py
artifacts/logs/audit.jsonl
```

**Critério de conclusão (Done when):**
- Perguntas fora do escopo retornam mensagem de recusa (testável manualmente)
- Toda resposta inclui disclaimer e, quando disponível, citação de fonte
- `artifacts/logs/audit.jsonl` cresce a cada interação com campos obrigatórios presentes

---

### M5 — Relatório Técnico, README e Vídeo Demo

**Objetivo:** Documentar a Fase 3 e preparar entregáveis finais para submissão.

**Tarefas:**

- [ ] Criar `Relatorio_Tecnico_Tech_Challenge_Fase3.md` com:
  - Descrição do dataset usado para fine-tuning (fonte, tamanho, processo de limpeza)
  - Arquitetura do fine-tuning (modelo base, parâmetros LoRA, hardware)
  - Resultados comparativos (modelo base vs. fine-tunado)
  - Arquitetura do assistente (diagrama do LangGraph flow)
  - Mecanismos de segurança implementados
  - Limitações e próximos passos
- [ ] Atualizar `README.md` com:
  - Instruções de instalação dos novos requisitos (`pip install -r requirements.txt`)
  - Comandos para rodar fine-tuning, assistente e auditoria
  - Estrutura de diretórios atualizada
- [ ] Gravar vídeo demo (5–10 min) mostrando:
  - Fine-tuning executando (ou logs do treino)
  - Assistente respondendo perguntas médicas com citações
  - Guard-rail bloqueando pergunta fora do escopo
  - Log de auditoria gerado
- [ ] Atualizar `requirements.txt` com todas as novas dependências

**Arquivos envolvidos:**
```
Relatorio_Tecnico_Tech_Challenge_Fase3.md
README.md
requirements.txt
```

**Critério de conclusão (Done when):**
- Relatório técnico cobre todos os 4 itens obrigatórios do challenge
- README permite reproduzir o pipeline completo (Fases 1+2+3) com um desenvolvedor novo
- Link do vídeo incluído no README

---

## Datasets Sugeridos

| Dataset | Tamanho | Licença | URL |
|---------|---------|---------|-----|
| MedQuAD | 47k pares Q&A | Public Domain | github.com/abachaa/MedQuAD |
| PubMedQA | 1k pares anotados | MIT | pubmedqa.github.io |
| MIMIC-III (notas clínicas) | Grande | PhysioNet (requer credencial) | physionet.org |
| Sintético via Gemini | Personalizado | — | Geração local com prompts |

**Recomendação:** Para velocidade de implementação, combinar MedQuAD (filtrado para oncologia) com exemplos sintéticos gerados pelo Gemini usando `LLM_USE_MOCK=false`.

---

## Estrutura de Diretórios — Fase 3

```
fiap-ia-challenge-phase-2/
├── data/
│   ├── finetune/
│   │   ├── train.jsonl
│   │   ├── val.jsonl
│   │   └── README.md
│   └── kb/                          # Knowledge base para RAG
├── src/
│   ├── assistant/
│   │   ├── __init__.py
│   │   ├── retriever.py
│   │   ├── chain.py
│   │   ├── graph.py
│   │   ├── tools.py
│   │   ├── guardrails.py
│   │   ├── audit_logger.py
│   │   └── explainer.py
│   └── llm/                         # Existente — reutilizar cliente Gemini
├── scripts/
│   ├── prepare_finetune_data.py
│   ├── fine_tune.py
│   ├── eval_finetune.py
│   └── run_assistant.py
├── artifacts/
│   ├── finetune_checkpoints/
│   │   └── final_adapter/
│   ├── vectorstore/
│   └── logs/
│       ├── finetune.jsonl
│       └── audit.jsonl
├── Relatorio_Tecnico_Tech_Challenge_Fase3.md
├── requirements.txt                 # Atualizado com novas deps
└── README.md                        # Atualizado
```

---

## Checklist de Entregáveis — Challenge Spec

### Repositório Git

- [ ] Repositório público no GitHub com histórico de commits organizado
- [ ] `README.md` com instruções completas de reprodução
- [ ] `requirements.txt` atualizado

### Entregáveis Técnicos Obrigatórios

- [ ] **Fine-tuning de LLM:** script funcional + checkpoint salvo + logs de treino
- [ ] **Pipeline LangChain/LangGraph:** assistente conversacional médico com RAG + fluxo de estados
- [ ] **Segurança e auditoria:** guard-rails + citações + `audit.jsonl` com todos os campos
- [ ] **Relatório técnico Fase 3:** documento em Markdown cobrindo metodologia, resultados e limitações

### Vídeo Demo

- [ ] Duração: 5–10 minutos
- [ ] Demonstra fine-tuning (resultado ou execução)
- [ ] Demonstra assistente respondendo com citação de fonte
- [ ] Demonstra guard-rail em ação
- [ ] Link do vídeo no `README.md`

---

## Dependências a Adicionar em `requirements.txt`

```
# Fine-tuning
transformers>=4.40.0
peft>=0.10.0
trl>=0.8.0
bitsandbytes>=0.43.0
accelerate>=0.29.0
datasets>=2.19.0

# LangChain / LangGraph
langchain>=0.2.0
langgraph>=0.1.0
langchain-community>=0.2.0
langchain-huggingface>=0.0.3

# Vector store
chromadb>=0.5.0
faiss-cpu>=1.8.0

# Anonimização (opcional)
presidio-analyzer>=2.2.0
presidio-anonymizer>=2.2.0
```

---

## Ordem de Implementação Recomendada

```
M1 (dados) → M2 (fine-tuning) → M3 (assistente) → M4 (segurança) → M5 (docs)
```

M3 pode começar em paralelo com M2 usando o modelo base (sem fine-tuning) para validar o pipeline LangChain/LangGraph antes do adapter estar pronto. Substituir o modelo ao final de M2.
