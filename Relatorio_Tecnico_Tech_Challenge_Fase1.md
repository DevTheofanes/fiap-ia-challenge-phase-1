# Relatório Técnico — Tech Challenge (Fase 1)

## 1) Estratégias de pré-processamento

**Base e alvo**
- CSV: `../data/wisconsin_breast_cancer.csv`
- Amostras: **569** | Features numéricas: **30**
- Alvo: `diagnosis` mapeado para `target` (`M→1`, `B→0`)
- Distribuição do alvo: **B=357 (62,8%)**, **M=212 (37,2%)**

**Limpeza e tipos**
- Remoção de colunas de identificação (ex.: `id`, `ID number`, `Unnamed: 32`)
- Numéricas mantidas nas famílias `*_mean`, `*_se`, `*_worst`

**Faltantes**
- Estratégia: imputação **mediana** por coluna numérica (robusta a outliers e reproduzível)

**Escalonamento**
- `StandardScaler` nas pipelines que requerem normalização (LR, SVC, KNN)

**Particionamento e avaliação**
- Split estratificado **60/20/20** (treino/validação/teste)
- **CV 5-fold estratificada** no treino para comparação de modelos
- **Validação** para seleção/tuning; **teste** preservado para avaliação final

**Limiar e custo de erro**
- Classe positiva: **maligno (1)**
- Ajuste de limiar visando **maior recall** (FN têm custo clínico mais alto)
- Calibração avaliada por **curva** e **Brier score**

---

## 2) Modelos usados e justificativa

**Candidatos**: Regressão Logística (LR), SVC (RBF), Random Forest, KNN (XGBoost opcional).

**Modelo vencedor**: **LR** (pipeline: `StandardScaler` + `LogisticRegression(class_weight='balanced', max_iter=2000, solver='lbfgs', random_state=42)`)
- **Por quê**: baseline forte em tabulares, rápido, estável e interpretável; com `class_weight='balanced'` lida bem com desbalanceamento moderado e, neste dataset, alcançou melhor equilíbrio **Recall ↔ Precisão** após ajuste de limiar.
- **Hiperparâmetros finais (principais)**: `C=1.0`, regularização L2 (padrão), `max_iter=2000`.

---

## 3) Resultados e interpretação

### 3.1 Validação (limiar 0,5)
- **Accuracy**: 0,9825  
- **Precision**: **1,0000**  
- **Recall (M)**: **0,9535**  
- **F1**: 0,9762  
- **ROC AUC**: **1,0000**

> Interpretação: na validação, a LR já apresentou **recall** elevado para maligno sem custo de precisão, sinal de boa separabilidade do conjunto.

### 3.2 Teste — antes e depois do ajuste de limiar

**Teste (limiar 0,5)**
- Accuracy: 0,9561  
- Precision: 0,9744  
- Recall (M): 0,9048  
- F1: 0,9383  
- ROC AUC: 0,9927  
- **Matriz**: TN=71, FP=1, FN=4, TP=38

**Limiar final escolhido**: **0,33** (critério F2/recall-priorizado)

**Teste (limiar 0,33)**
- Accuracy: **0,9737**  
- Precision: 0,9756  
- Recall (M): **0,9524**  
- F1: **0,9639**  
- **Matriz**: TN=71, FP=1, FN=2, TP=40

> Interpretação do trade-off: ao **reduzir o limiar** (0,5 → **0,33**), o **Recall (M)** subiu de **0,9048** para **0,9524** (reduzindo **FN** de 4→2), **sem** piorar a Precisão (permaneceu ≈0,976) e ainda **melhorando F1** (0,938→0,964). Para um cenário clínico, este é um ganho valioso: menos malignos passam despercebidos, com mínimo aumento de falsos positivos (FP continuou **1**).

### 3.3 Calibração
- **Brier score (validação)**: **0,0129** (baixo)  
> Probabilidades bem calibradas; dá confiança para usar pontuações como **escore de risco** e aplicar limiares clínicos.

### 3.4 Importância de features (Permutation Importance — top 10)
1. `texture_worst` (0,0397)  
2. `concave_points_mean` (0,0327)  
3. `concave_points_worst` (0,0279)  
4. `symmetry_worst` (0,0217)  
5. `radius_se` (0,0182)  
6. `concavity_worst` (0,0182)  
7. `radius_worst` (0,0166)  
8. `area_worst` (0,0164)  
9. `area_se` (0,0150)  
10. `concavity_mean` (0,0145)

> Leitura clínica: padrões de **concavidade/irregularidade** (`concave_points_*`, `concavity_*`) e medidas de **tamanho/forma** em piores casos (`*_worst`) figuram entre os principais preditores — consistente com a literatura do WBCD.

---

## 4) Conclusões

- A **LR** com `class_weight='balanced'` foi **competitiva** e estável.  
- O **ajuste de limiar** para **0,33** **melhorou** sensivelmente a **sensibilidade** para maligno (FN reduzidos de 4→2) **sem penalizar** a precisão — excelente perfil para triagem.  
- **Calibração** sólida (Brier baixo) permite tratar a saída como **probabilidade de malignidade**.  
- As **features de forma/concavidade e “worst”** sustentam a decisão do modelo, coerentes com diagnóstico por imagem.

---

## 5) Recomendações

1. **Manter limiar ajustado** (0,33) em produção, mas permitir **parametrização** por política clínica (ex.: meta de recall).  
2. Avaliar **calibração periódica** (drift): replotar curva e Brier ao incorporar dados novos.  
3. Registrar **matriz de confusão** e **métricas por período** (monitoramento de FN/FP).  
4. Se houver expansão de dados, experimentar **SVC/XGBoost** com tuning mais amplo e eventualmente **calibração isotônica/Platt**.  
5. Incorporar **explicações locais** (SHAP por amostra) no laudo para apoio ao especialista.

---

### Anexo: resumo bruto das métricas
- `best_model_name`: **LR**  
- `val@0.5`: acc=0,9825 | prec=1,0000 | rec=0,9535 | f1=0,9762 | roc_auc=1,0000  
- `test@0.5`: acc=0,9561 | prec=0,9744 | rec=0,9048 | f1=0,9383 | roc_auc=0,9927 | CM=[71,1,4,38]  
- `final_threshold`: **0,33**  
- `test@thr`: acc=0,9737 | prec=0,9756 | rec=0,9524 | f1=0,9639 | CM=[71,1,2,40]  
- `brier_val`: **0,0129**  
- `perm_importance_top10`: conforme lista acima
