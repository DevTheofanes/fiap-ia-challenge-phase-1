# Relatorio Tecnico - Tech Challenge Fase 2

## 1. Contexto e continuidade da Fase 1
Este relatorio consolida a Fase 2 do projeto de diagnostico de cancer de mama com o dataset Wisconsin Breast Cancer, dando continuidade direta ao pipeline da Fase 1. O baseline e os splits permanecem os mesmos e as metricas foram preservadas para comparacao justa.

Evidencias:
- Baseline e hiperparametros salvos em `artifacts/baseline_metrics.json`.
- Modelos baseline em `artifacts/baseline_models/`.

## 2. Objetivo da Fase 2
Aplicar Algoritmo Genetico (GA) para otimizar hiperparametros dos modelos principais (Logistic Regression e Random Forest), comparar com o baseline e integrar uma camada de LLM para explicacoes em linguagem natural.

## 3. Algoritmo Genetico (GA)

### 3.1 Codificacao e espaco de busca
Os hiperparametros foram codificados como genes continuos, discretos e categoricos. A definicao do espaco de busca e regras de validacao/reparo foram implementadas para manter individuos validos.

Arquivos principais:
- Codificacao e reparo: `src/genetic/encoding.py`
- Espaco de busca: `src/genetic/search_space.py`

### 3.2 Operadores geneticos
O GA utiliza:
- Selecao por torneio
- Crossover uniforme
- Mutacao por gene
- Elitismo explicito

Implementacao: `src/genetic/ga.py`

### 3.3 Funcao de fitness
A funcao fitness usa F1-score com validacao cruzada estratificada. Resultados de fitness sao cacheados para eficiencia.

Implementacao: `src/genetic/fitness.py`

## 4. Experimentos com GA

Foram executados tres conjuntos de configuracoes (A, B, C), variando populacao, taxa de mutacao, numero de geracoes e pressao de selecao.

Configuracoes:
- Definicoes em `scripts/run_ga_experiments.py`
- Logs e artefatos em `artifacts/ga_runs/<modelo>/exp<id>_seed<seed>/`

Reprodutibilidade:
- Seeds controladas por experimento
- Historico por geracao em `history.csv`
- Melhor individuo em `best.json`

## 5. Resultados e comparacao baseline vs GA

A comparacao foi feita mantendo o mesmo protocolo de avaliacao e splits. A tabela comparativa e as figuras estao consolidadas em:
- `reports/phase2_results.md`
- `reports/figures/`

Resumo:
- Para LR, o GA melhorou o F1 no holdout e elevou o recall, mantendo precisao alta.
- Para RF, o GA manteve desempenho semelhante ao baseline no holdout.

## 6. Integracao com LLM

Foi adicionada uma camada de explicacao em linguagem natural para casos clinicos, com saida estruturada em JSON e fallback quando a LLM nao estiver disponivel.

Componentes:
- Cliente e schemas: `src/llm/client.py`, `src/llm/schemas.py`
- Prompts versionados: `src/llm/prompts.py`
- Geracao de explicacoes: `src/llm/explain.py`

Exemplos e logs:
- Explicacoes: `artifacts/llm/sample_explanations.jsonl`
- Logs de interacao: `artifacts/llm/llm_logs.jsonl`

## 7. Avaliacao das explicacoes

Foi aplicada uma rubrica manual com criterios de clareza, consistencia, nao-alarmismo, utilidade clinica e conformidade. A tabela preenchida esta em:
- `reports/llm_eval.md`

## 8. Monitoramento e logging

Foram adotados logs estruturados em JSONL, separados por etapa:
- Treino: `artifacts/logs/training.jsonl`
- Avaliacao: `artifacts/logs/evaluation.jsonl`
- LLM: `artifacts/logs/llm.jsonl`

Utilitario de logging: `src/logging_utils.py`

## 9. Arquitetura e escalabilidade

A arquitetura segue fluxo:
Dados -> Preprocessamento -> Baseline/GA -> Avaliacao -> LLM -> Artefatos/Logs

A escalabilidade e a possibilidade de paralelizacao do fitness foram discutidas e suportadas por:
- Execucoes independentes por seed/experimento
- Possivel paralelizacao de avaliacao com joblib/multiprocessing

Resumo no `README.md`.

## 10. Reprodutibilidade

Como reproduzir:
1. `python scripts/run_baseline.py`
2. `python scripts/run_ga_experiments.py --model LR --exp A --seed 42` (variar experimento e modelo)
3. `python scripts/summarize_ga_runs.py`
4. `python scripts/summarize_results.py`
5. `python scripts/explain_sample.py --split val --sample-idx 0`

Dependencias: `requirements.txt`

## 11. Limitacoes

- Dataset pequeno e sensivel ao split.
- Espaco de busca limitado para manter custo computacional viavel.
- Avaliacao das explicacoes e manual, dependente de julgamento humano.

## 12. Conclusao

A Fase 2 entregou a otimizacao via GA, a comparacao cientifica com o baseline e a integracao com LLM para explicacoes clinicas, com logging estruturado e artefatos completos para auditoria.
