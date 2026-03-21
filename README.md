# FIAP IA Challenge Phase 1 – Breast Cancer Diagnosis Pipeline

Este repositório contém a solução desenvolvida para o Tech Challenge (Fase 1) da FIAP, focado
na construção de um pipeline de aprendizado de máquina para diagnóstico do câncer de mama a
partir do conjunto de dados de Wisconsin Breast Cancer.

## Visão geral

O script principal (`src/main.py`) executa um fluxo completo de experimentação:

1. **Carregamento e exploração dos dados** – inspeção de formato, estatísticas básicas e
   distribuição da variável alvo.
2. **Pré-processamento** – limpeza de identificadores, tratamento de valores ausentes e
   padronização.
3. **Treinamento e avaliação** – comparação de modelos clássicos de classificação com
   validação cruzada, ajuste de hiperparâmetros e seleção baseada em desempenho no conjunto
   de validação.
4. **Métricas avançadas** – geração de curvas ROC/PR, matriz de confusão, curva de
   aprendizado, calibração e análise de limiar.
5. **Explicabilidade** – cálculo opcional de importâncias por SHAP (quando disponível) e
   permutation importance como fallback.
6. **Persistência** – salvamento do melhor modelo, lista de features e limiar ótimo em um
   arquivo `best_model_with_threshold.joblib`.

Todos os gráficos são renderizados utilizando o backend `Agg` do Matplotlib, o que permite a
execução em ambientes headless (como containers Docker) sem dependências gráficas extras.

## Estrutura do projeto

```
.
├── data/                          # Arquivos de dados brutos
├── notebooks/                     # Explorações e estudos em Jupyter
├── src/
│   └── main.py                    # Pipeline completo de treinamento
│   └── genetic/                   # Espaço de busca e codificação genética
├── requirements.txt               # Dependências Python
├── Dockerfile                     # Imagem para execução containerizada
└── README.md
```

## Pré-requisitos

- Python 3.10 ou superior.
- Dependências listadas em `requirements.txt`.
- Arquivo de dados `wisconsin_breast_cancer.csv` disponível em `data/`.

Dependências opcionais:

- `shap` e `xgboost` habilitam cálculos adicionais de interpretabilidade e um algoritmo extra
  durante o treinamento. Caso não estejam instalados, o pipeline continua funcional, apenas
  pulando essas etapas.

## Instalação e execução local

1. Crie e ative um ambiente virtual (opcional, porém recomendado):

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Linux/macOS
   .venv\Scripts\activate     # Windows PowerShell
   ```

2. Instale as dependências:

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

3. (Opcional) instale dependências extras para interpretabilidade:

   ```bash
   pip install shap xgboost
   ```

4. Execute o pipeline:

   ```bash
   python src/main.py
   ```

Os resultados numéricos são impressos no terminal. Gráficos são exibidos em modo não
interativo; para salvá-los, configure `matplotlib` conforme necessário no próprio script.

## Execução com Docker

1. Construa a imagem:

   ```bash
   docker build -t fiap-ia-challenge .
   ```

2. Execute o container:

   ```bash
   docker run --rm -v "$(pwd)/data:/app/data" fiap-ia-challenge
   ```

   O volume garante que o dataset local seja montado dentro do container. Ao final da
   execução o arquivo `best_model_with_threshold.joblib` ficará disponível dentro do
   container em `/app`; mapeie um volume adicional caso queira persistir o artefato no host.

## Relatório técnico

Detalhes completos sobre escolhas de modelagem e resultados podem ser encontrados em
[`Relatorio_Tecnico_Tech_Challenge_Fase1.md`](docs/reports/Relatorio_Tecnico_Tech_Challenge_Fase1.md).

## Genome design & constraints

- Modelos alvo: `LR` (Logistic Regression) e `RF` (Random Forest), otimizados separadamente.
- Métrica fitness: `f1` (definida em `src/config.py`); avaliação recomendada com
  `StratifiedKFold` de 3 folds no conjunto de treino.
- Espaço de busca: definido em `src/genetic/search_space.py`.
- Codificação: funções `random_individual`, `mutate`, `crossover`, `decode`, `repair` em
  `src/genetic/encoding.py` (genes contínuos, discretos e categóricos).
- Restrições tratadas no `repair()`:
  - `penalty`/`solver` válidos em Logistic Regression.
  - `min_samples_split > min_samples_leaf` e `criterion` suportado em Random Forest.

## Integração com LLM para interpretação

Adiciona uma camada de LLM ao pipeline para explicar diagnósticos e resumir resultados
de experimentos em linguagem natural, com saída estruturada em JSON e fallback templateado.

### Principais recursos

- **Contrato de entrada/saída** para explicações clínicas e resumos de métricas.
- **Prompts versionados** em `src/llm/prompts.py`.
- **Validação do JSON** (schemas) em `src/llm/schemas.py`.
- **Logging** de prompts e respostas em `artifacts/llm/llm_logs.jsonl` (com redaction).
- **Fallback** quando a LLM não estiver configurada.

### Estrutura adicionada

```
src/llm/
  client.py          # Cliente Gemini (ou mock) e carregamento de .env
  prompts.py         # Templates dos prompts
  schemas.py         # Parsers e validação de JSON
  explain.py         # Geração de explicação por amostra
  summarize.py       # Resumo de métricas/experimentos
scripts/
  explain_sample.py  # Roda explicação em 1 amostra
  summarize_results.py # Resume baseline vs GA
artifacts/llm/
  sample_explanations.jsonl
  experiment_summaries.md
docs/ai/
  llm_eval.md        # Rubrica de avaliação manual
```

### Configuração do Gemini

1. Copie/ajuste o arquivo `.env` com sua chave:

   ```bash
   GEMINI_API_KEY=...
   GEMINI_MODEL=gemini-1.5-flash
   ```

2. Instale a dependência:

   ```bash
   pip install -r requirements.txt
   ```

Se a chave não estiver definida, o pipeline gera respostas templateadas automaticamente.

### Exemplos de uso

- Explicar uma amostra (salva JSONL e log):

  ```bash
  python scripts/explain_sample.py --split val --sample-idx 0
  ```

- Resumir baseline vs GA (gera `artifacts/llm/experiment_summaries.md`):

  ```bash
  python scripts/summarize_results.py
  ```

### Avaliação de qualidade (rubrica)

Preencha a tabela em `docs/ai/llm_eval.md` com 10 amostras, avaliando clareza, coerência,
não-alarmismo, utilidade clínica e conformidade.

## Monitoramento, Logging e Escalabilidade

### Logging estruturado
Os logs ficam em `artifacts/logs/` no formato JSONL, com uma linha por evento. Campos
chave incluem timestamp, stage, model, experiment, seed, metricas finais e tempo de execucao.

Arquivos gerados:
- `training.jsonl` (baseline e GA)
- `evaluation.jsonl` (metricas por split)
- `llm.jsonl` (eventos de explicabilidade)

Exemplo de linha:
```
{"timestamp":"2026-01-20T18:42:10+00:00","level":"INFO","message":"ga_train","stage":"ga_train","model":"RF","experiment":"expB","seed":42,"best_f1":0.94,"duration_sec":312.5}
```

### Tracking de experimentos (manual)
Cada experimento gera:
- `history.csv` com convergencia do GA
- `best.json` com melhor individuo e metricas
- `best_model.joblib`
Os agregados ficam em `artifacts/ga_summary/*.csv` via `scripts/summarize_ga_runs.py`.

### Monitoramento de performance
Os logs de treinamento registram:
- tempo total por experimento
- tempo medio por geracao (`mean_gen_time_sec`)
- numero de avaliacoes (`eval_count`)
- tempo medio por avaliacao (`mean_eval_time_sec`)
Esses dados alimentam a discussao de custo computacional.

### Preparacao para escalabilidade (conceitual)
- Avaliacao de fitness e independente, podendo ser paralelizada com `multiprocessing` ou `joblib`.
- Seeds/experimentos rodam como jobs isolados, permitindo escala horizontal.
- Docker garante reprodutibilidade e isolamento do ambiente.

### Arquitetura (diagrama simples)
```
Dataset
  -> Data Loader
    -> Preprocessing
      -> Baseline Models
      -> GA Optimizer
        -> Best Model
          -> Evaluation
            -> LLM Interpreter
              -> Artifacts/Logs
```

## Contato

Dúvidas ou sugestões podem ser direcionadas via issues neste repositório.
