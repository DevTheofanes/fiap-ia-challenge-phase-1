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
[`Relatorio_Tecnico_Tech_Challenge_Fase1.md`](Relatorio_Tecnico_Tech_Challenge_Fase1.md).

## Contato

Dúvidas ou sugestões podem ser direcionadas via issues neste repositório.
