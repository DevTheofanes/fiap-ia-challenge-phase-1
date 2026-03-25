# Script do Vídeo — Tech Challenge FIAP IADT Fase 3
> Duração estimada: 12–14 minutos

---

## CENA 1 — Abertura (0:00 – 1:00)

**[Tela: slide ou terminal com o título do projeto]**

> "Olá! Neste vídeo vou apresentar o Tech Challenge da Fase 3 do programa de pós-graduação em Inteligência Artificial da FIAP.
>
> O desafio desta fase foi criar um assistente virtual médico treinado com dados hospitalares, capaz de auxiliar médicos em condutas clínicas, responder dúvidas contextualizadas e coordenar fluxos automatizados com LangChain e LangGraph.
>
> Vou cobrir quatro grandes blocos: o pipeline de fine-tuning do modelo, a arquitetura do assistente, a execução de um fluxo completo com perguntas clínicas reais, e os logs de auditoria e validação."

---

## CENA 2 — Estrutura do Projeto (1:00 – 2:00)

**[Tela: `tree` ou explorador mostrando a estrutura de pastas]**

```
scripts/          ← pipelines executáveis
src/assistant/    ← núcleo do assistente
data/finetune/    ← dataset de treino
data/patients/    ← prontuários sintéticos
data/kb/          ← base de conhecimento
artifacts/        ← checkpoints, vectorstore, logs
```

> "O projeto é totalmente modularizado em Python. Os scripts ficam separados da lógica de negócio em `src/`. Os dados de treinamento, os prontuários sintéticos dos pacientes e a base de conhecimento médica ficam em `data/`. Os artefatos gerados — checkpoints do modelo, a vectorstore e os logs — ficam em `artifacts/`."

---

## CENA 3 — Fine-Tuning: Dataset (2:00 – 4:00)

**[Tela: abrir `data/finetune/README.md` ou mostrar head do `train.jsonl`]**

> "Começando pelo fine-tuning. O dataset foi construído a partir de fontes públicas — principalmente o MedQuAD e exemplos sintéticos de oncologia — e passou por três etapas de curadoria."

**[Tela: `scripts/prepare_finetune_data.py` — mostrar as funções de filtragem, dedup e anonimização]**

> "Primeiro, filtragem por relevância oncológica. Segundo, deduplicação por pergunta para evitar overfitting. Terceiro, anonimização por expressões regulares — datas, IDs, nomes, telefones e e-mails foram substituídos por tokens genéricos."

**[Tela: mostrar um exemplo do `train.jsonl`]**

> "O resultado são 2.456 pares de instrução para treino e 614 para validação, todos no formato instruction/input/output compatível com o SFTTrainer."

---

## CENA 4 — Fine-Tuning: Treinamento (4:00 – 5:30)

**[Tela: `scripts/fine_tune.py` — mostrar config LoRA e chamada ao SFTTrainer]**

> "O modelo base é o TinyLlama 1.1B Chat. Escolhemos ele por ser leve o suficiente para rodar localmente, mantendo capacidade de seguir instruções médicas.
>
> O ajuste fino usa LoRA — Low-Rank Adaptation — com rank 8 e alpha 16. Isso significa que treinamos apenas uma fração pequena dos parâmetros, o que reduz drasticamente memória e tempo de treinamento sem sacrificar qualidade.
>
> O treinamento é supervisionado via `trl.SFTTrainer` e salva checkpoints a cada época."

**[Tela: mostrar `artifacts/logs/finetune.jsonl` — eventos de training_start e training_done]**

> "Aqui nos logs vemos os eventos de início e conclusão do treinamento, com parâmetros registrados para reprodutibilidade."

**[Tela: `scripts/eval_finetune.py` ou mostrar trecho de saída com scores ROUGE]**

> "O script de avaliação compara o modelo base com o fine-tuned usando ROUGE, mostrando a melhora nas respostas médicas após o ajuste."

---

## CENA 5 — Arquitetura do Assistente (5:30 – 7:30)

**[Tela: diagrama do fluxo LangGraph — copiar do relatório técnico]**

```
classify_intent → retrieve_kb_context → retrieve_patient_context
                                                    ↓
                                          generate_response
                                                    ↓
                                          validate_response
          ↓ (out_of_scope)
       refuse_response
```

> "O assistente é orquestrado por um grafo LangGraph com 7 nós. Vou explicar cada um."

**[Tela: `src/assistant/graph.py` — mostrar as funções dos nós]**

> "`classify_intent` analisa a pergunta e decide se é médica ou fora de escopo. Perguntas sobre clima, criptomoedas ou qualquer tema não clínico são recusadas aqui.
>
> `retrieve_kb_context` busca os 3 documentos mais relevantes na vectorstore ChromaDB, indexada com os 729 Q&As do MedQuAD.
>
> `retrieve_patient_context` carrega o prontuário sintético do paciente, se um ID foi fornecido.
>
> `generate_response` usa o modelo TinyLlama fine-tunado localmente para gerar a resposta com base no prompt que combina pergunta, contexto da KB e dados do paciente.
>
> `validate_response` aplica os guardrails, adiciona a explicabilidade e o disclaimer obrigatório."

---

## CENA 6 — Demo: Fluxo Automatizado com Paciente (7:30 – 10:00)

**[Tela: terminal — executar o assistente com paciente]**

```bash
python scripts/run_assistant.py --patient-id P-0001
```

> "Vou agora executar um fluxo completo. Estou iniciando o assistente com o prontuário da paciente P-0001 — uma mulher de 54 anos com uma lesão BI-RADS 4 e probabilidade de malignidade de 81% prevista pelo nosso modelo de ML das fases anteriores."

**[Tela: digitar a pergunta]**

```
Pergunta: Quais são os próximos passos recomendados para uma lesão BI-RADS 4?
```

> "Faço uma pergunta clínica sobre o próximo passo para essa classificação."

**[Tela: mostrar a resposta gerada com as citações de fontes]**

> "O assistente retorna uma resposta contextualizada com as informações do prontuário e cita explicitamente os documentos da base de conhecimento usados — aqui você vê as referências ao MedQuAD e ao registro da paciente. Isso atende ao requisito de explainability: o médico sabe exatamente de onde veio cada informação.
>
> Note também o disclaimer ao final: o assistente nunca prescreve diretamente nem emite diagnóstico definitivo sem validação humana."

**[Tela: digitar uma segunda pergunta fora do escopo]**

```
Pergunta: Qual é a previsão do tempo para amanhã?
```

> "Agora vou testar o guardrail com uma pergunta fora do escopo médico."

**[Tela: mostrar a recusa do assistente]**

> "O nó `classify_intent` identifica que a pergunta não é médica e o fluxo desvia para `refuse_response`. O assistente recusa educadamente e orienta o usuário a perguntar sobre saúde."

---

## CENA 7 — Demo: Guardrail de Prescrição (10:00 – 11:00)

**[Tela: terminal — nova pergunta]**

```
Pergunta: Pode me prescrever tamoxifeno 20mg por dia?
```

> "Vou testar o guardrail mais crítico: pedido de prescrição direta."

**[Tela: mostrar a recusa]**

> "O filtro de output detecta a intenção de prescrição e bloqueia a resposta antes de ela chegar ao usuário, substituindo pelo aviso de que o assistente não pode prescrever medicamentos diretamente — essa decisão exige um médico responsável."

---

## CENA 8 — Logs e Auditoria (11:00 – 12:30)

**[Tela: abrir `artifacts/logs/audit.jsonl` — mostrar um registro formatado]**

> "Toda interação gera um evento no log de auditoria. Aqui vemos o registro completo: timestamp, a pergunta do usuário, os documentos recuperados, a resposta gerada pelo modelo, se o guardrail foi acionado, a intenção classificada e o ID do paciente."

```json
{
  "timestamp": "...",
  "user_query": "Quais são os próximos passos...",
  "intent": "medical",
  "patient_id": "P-0001",
  "patient_context_used": true,
  "kb_sources": ["0000001_7_0007.txt", "..."],
  "guardrail_triggered": false,
  "model_response": "..."
}
```

> "Esse trail de auditoria atende diretamente ao requisito de logging detalhado para rastreamento — qualquer resposta do assistente pode ser auditada, reproduzida e contestada."

**[Tela: mostrar `artifacts/logs/assistant.jsonl`]**

> "Além do audit log, temos o log operacional do assistente com todos os eventos de execução do grafo — útil para debug e monitoramento de produção."

---

## CENA 9 — Encerramento (12:30 – 13:30)

**[Tela: diagrama final ou slide de resumo]**

> "Para resumir o que foi entregado nesta fase:
>
> — Um pipeline completo de fine-tuning com LoRA sobre TinyLlama, treinado em 3.070 pares médicos curados e anonimizados.
>
> — Um assistente médico com fluxo LangGraph de 7 nós integrando RAG sobre base de conhecimento pública, prontuários estruturados de pacientes e o modelo fine-tunado como gerador principal.
>
> — Guardrails em duas camadas: filtro de input para classificar intenção, e filtro de output para bloquear prescrições diretas e suavizar diagnósticos definitivos.
>
> — Explainability com citação explícita de fontes em toda resposta.
>
> — Audit log completo de toda interação para rastreabilidade.
>
> O código está disponível no repositório com README completo de instalação e execução. Obrigado!"

---

## Checklist de Gravação

- [ ] Ambiente ativo: `source .venv/bin/activate`
- [ ] `.env` configurado com `GEMINI_API_KEY` (ou `LLM_USE_MOCK=true` para demo sem API)
- [ ] Vectorstore já construída: `python scripts/build_kb.py` (se necessário)
- [ ] Adapter fine-tunado presente: `artifacts/finetune_checkpoints/final_adapter/`
- [ ] Paciente P-0001 presente: `data/patients/P-0001.json`
- [ ] Terminal com fonte grande e tema escuro para boa legibilidade
- [ ] Gravação em 1080p mínimo
- [ ] Duração dentro de 15 minutos

---

## Comandos de Demo (copiar e colar durante a gravação)

```bash
# Iniciar assistente com paciente
python scripts/run_assistant.py --patient-id P-0001

# Perguntas para demo
# 1. Fluxo normal (médica + paciente):
Quais são os próximos passos recomendados para uma lesão BI-RADS 4?

# 2. Fora do escopo:
Qual é a previsão do tempo para amanhã?

# 3. Guardrail de prescrição:
Pode me prescrever tamoxifeno 20mg por dia?

# 4. Pergunta sobre diagnóstico (verifica suavização):
Esta paciente tem câncer?

# Ver logs em tempo real
tail -f artifacts/logs/audit.jsonl | python -m json.tool
```
