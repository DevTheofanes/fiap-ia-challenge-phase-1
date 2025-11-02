
# === Configurações iniciais ===
from __future__ import annotations

# Caminho para o CSV
DATA_PATH = '../data/wisconsin_breast_cancer.csv'

# Coluna alvo esperada no dataset (diagnóstico: 'M' ou 'B')
TARGET_COL = 'diagnosis'

# Colunas de identificação (removidas das features)
ID_COLS = ['id', 'ID number', 'Unnamed: 32']  # ajuste conforme seu arquivo

RANDOM_STATE = 42



# === Imports ===
import os, io, warnings, joblib
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from typing import Dict, List, Tuple

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix, RocCurveDisplay, PrecisionRecallDisplay
)
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

try:
    import shap  # type: ignore
    HAS_SHAP = True
except Exception:
    HAS_SHAP = False

try:
    from xgboost import XGBClassifier  # type: ignore
    HAS_XGB = True
except Exception:
    HAS_XGB = False



# === Leitura ===
assert os.path.exists(DATA_PATH), f"Arquivo CSV não encontrado: {DATA_PATH}"
df = pd.read_csv(DATA_PATH)
print("Formato:", df.shape)
display(df.head())



# Info seguro
buf = io.StringIO()
df.info(buf=buf)
print(buf.getvalue())



# Estatísticas descritivas (numéricas)
display(df.describe().T)



# Missing values
missing = df.isna().sum().sort_values(ascending=False)
display(missing[missing > 0])



# Distribuição da variável alvo
target_counts = df[TARGET_COL].value_counts(dropna=False)
print(target_counts)
target_counts.plot(kind='bar', title='Distribuição da variável alvo (diagnosis)')
plt.xlabel('Classe'); plt.ylabel('Contagem'); plt.show()



# Limpeza básica
cols_to_drop = [c for c in ID_COLS if c in df.columns]
df = df.drop(columns=cols_to_drop, errors='ignore')
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

for col in num_cols:
    if df[col].isna().any():
        df[col] = df[col].fillna(df[col].median())

# Mapeamento do alvo
df['target'] = df[TARGET_COL].map({'M': 1, 'B': 0})
assert set(df['target'].dropna().unique()).issubset({0,1}), "Mapeamento do alvo falhou."



# Correlações com o alvo
corr = df.corr(numeric_only=True)
target_corr = corr['target'].sort_values(ascending=False)
display(target_corr.head(15)); display(target_corr.tail(15))



# Heatmap simples (matplotlib)
N = 12
target_abs = df.corr(numeric_only=True)['target'].abs().sort_values(ascending=False)
top_features = target_abs.index[1:N+1]
subcorr = df[top_features].corr(numeric_only=True).values

fig, ax = plt.subplots(figsize=(6,5))
im = ax.imshow(subcorr, interpolation='nearest')
ax.set_xticks(range(len(top_features))); ax.set_yticks(range(len(top_features)))
ax.set_xticklabels(top_features, rotation=90); ax.set_yticklabels(top_features)
ax.set_title('Correlação entre Top Features'); fig.colorbar(im); plt.tight_layout(); plt.show()



# Seleção de features numéricas e alvo
feature_cols = [c for c in df.columns if c not in [TARGET_COL, 'target']]
X = df[feature_cols].select_dtypes(include=[np.number])
y = df['target']

# Split 60/20/20
X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.4, stratify=y, random_state=RANDOM_STATE)
X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=RANDOM_STATE)

print({k:v.shape for k,v in {'X_train':X_train,'X_val':X_val,'X_test':X_test}.items()})
print({k:float(v.mean()) for k,v in {'y_train':y_train,'y_val':y_val,'y_test':y_test}.items()})



# Baseline: Regressão Logística
baseline = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(max_iter=2000, class_weight='balanced', random_state=RANDOM_STATE))
])
baseline.fit(X_train, y_train)

def eval_and_print(name, y_true, y_pred, y_prob=None):
    print(f"== {name} ==")
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred))
    print("Recall:", recall_score(y_true, y_pred))
    print("F1:", f1_score(y_true, y_pred))
    if y_prob is not None:
        try:
            print("ROC AUC:", roc_auc_score(y_true, y_prob))
        except Exception as e:
            print("ROC AUC: n/d", e)
    print(classification_report(y_true, y_pred, digits=4))

# Val e Teste
y_val_pred = baseline.predict(X_val)
y_val_prob = baseline.predict_proba(X_val)[:,1]
eval_and_print("Validação (Baseline LR)", y_val, y_val_pred, y_val_prob)

y_test_pred = baseline.predict(X_test)
y_test_prob = baseline.predict_proba(X_test)[:,1]
eval_and_print("Teste (Baseline LR)", y_test, y_test_pred, y_test_prob)

RocCurveDisplay.from_predictions(y_val, y_val_prob)
plt.title("ROC — Validação (Baseline LR)"); plt.show()

PrecisionRecallDisplay.from_predictions(y_val, y_val_prob)
plt.title("Precision-Recall — Validação (Baseline LR)"); plt.show()



models = {
    'LR': Pipeline([('scaler', StandardScaler()),
                    ('clf', LogisticRegression(max_iter=2000, class_weight='balanced', random_state=RANDOM_STATE))]),
    'SVC': Pipeline([('scaler', StandardScaler()),
                     ('clf', SVC(probability=True, class_weight='balanced', random_state=RANDOM_STATE))]),
    'RF': Pipeline([('clf', RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, class_weight='balanced'))]),
    'KNN': Pipeline([('scaler', StandardScaler()),
                     ('clf', KNeighborsClassifier(n_neighbors=9))]),
}
if HAS_XGB:
    models['XGB'] = Pipeline([('clf', XGBClassifier(
        n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9,
        reg_lambda=1.0, n_jobs=4, random_state=RANDOM_STATE, eval_metric='logloss'
    ))])

# Validação cruzada nos dados de treino
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
cv_results = []
for name, pipe in models.items():
    acc = cross_val_score(pipe, X_train, y_train, scoring='accuracy', cv=cv)
    f1 = cross_val_score(pipe, X_train, y_train, scoring='f1', cv=cv)
    rec = cross_val_score(pipe, X_train, y_train, scoring='recall', cv=cv)
    cv_results.append({'model': name, 'cv_accuracy_mean': acc.mean(), 'cv_f1_mean': f1.mean(), 'cv_recall_mean': rec.mean()})
cv_df = pd.DataFrame(cv_results).sort_values(by=['cv_f1_mean','cv_recall_mean'], ascending=False)
display(cv_df)



# Espaços de busca enxutos
param_grids = {
    'LR': {'clf__C': [0.1, 1.0, 3.0]},
    'SVC': {'clf__C': [0.5, 1.0, 2.0], 'clf__gamma': ['scale', 0.1, 0.01]},
    'RF': {'clf__n_estimators': [200, 400], 'clf__max_depth': [None, 4, 8]},
    'KNN': {'clf__n_neighbors': [5, 9, 15]}
}
if HAS_XGB:
    param_grids['XGB'] = {'clf__n_estimators': [200, 400], 'clf__max_depth': [3, 4, 5], 'clf__learning_rate': [0.05, 0.1]}

best_models = {}
for name, base in models.items():
    grid = GridSearchCV(base, param_grids.get(name, {}), scoring='f1', cv=3, n_jobs=-1, refit=True)
    grid.fit(X_train, y_train)
    best_models[name] = grid.best_estimator_
    print(f"{name} -> best params:", grid.best_params_)



# Seleção com base no conjunto de validação
def metrics_table(model_name, y_true, y_pred, y_prob):
    return {
        'model': model_name,
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred),
        'roc_auc': roc_auc_score(y_true, y_prob) if y_prob is not None else np.nan,
    }

val_rows = []
for name, mdl in best_models.items():
    y_pred = mdl.predict(X_val)
    y_prob = mdl.predict_proba(X_val)[:,1] if hasattr(mdl, "predict_proba") else None
    val_rows.append(metrics_table(name, y_val, y_pred, y_prob))

val_df = pd.DataFrame(val_rows).sort_values(by=['f1','recall','accuracy'], ascending=False)
display(val_df)
best_name = val_df.iloc[0]['model']
print("Melhor modelo pela validação:", best_name)
best_estimator = best_models[best_name]



# Teste final do melhor modelo (padrão limiar 0.5)
if hasattr(best_estimator, "predict_proba"):
    y_test_prob_best = best_estimator.predict_proba(X_test)[:,1]
else:
    if hasattr(best_estimator, "decision_function"):
        scores = best_estimator.decision_function(X_test)
        smin, smax = scores.min(), scores.max()
        y_test_prob_best = (scores - smin) / (smax - smin + 1e-9)
    else:
        y_test_prob_best = None

y_test_pred_best = best_estimator.predict(X_test)

print("== TESTE (melhor modelo, limiar 0.5) ==")
print("Accuracy:", accuracy_score(y_test, y_test_pred_best))
print("Precision:", precision_score(y_test, y_test_pred_best))
print("Recall:", recall_score(y_test, y_test_pred_best))
print("F1:", f1_score(y_test, y_test_pred_best))
if y_test_prob_best is not None:
    print("ROC AUC:", roc_auc_score(y_test, y_test_prob_best))
print("\nClassification Report (teste):\n", classification_report(y_test, y_test_pred_best, digits=4))

cm = confusion_matrix(y_test, y_test_pred_best)
fig, ax = plt.subplots()
im = ax.imshow(cm)
ax.set_title(f'Matriz de Confusão — Teste ({best_name}, limiar 0.5)')
ax.set_xlabel('Predito'); ax.set_ylabel('Verdadeiro')
ax.set_xticks([0,1]); ax.set_yticks([0,1])
ax.set_xticklabels(['Benigno (0)', 'Maligno (1)']); ax.set_yticklabels(['Benigno (0)', 'Maligno (1)'])
for (i, j), z in np.ndenumerate(cm):
    ax.text(j, i, str(z), ha='center', va='center')
fig.colorbar(im); plt.show()

RocCurveDisplay.from_predictions(y_test, y_test_prob_best)
plt.title(f"ROC — Teste ({best_name})"); plt.show()

PrecisionRecallDisplay.from_predictions(y_test, y_test_prob_best)
plt.title(f"Precision-Recall — Teste ({best_name})"); plt.show()



train_sizes, train_scores, val_scores = learning_curve(
    best_estimator, X_train, y_train, cv=5, scoring='f1',
    train_sizes=np.linspace(0.1, 1.0, 5), n_jobs=-1
)
train_mean = train_scores.mean(axis=1)
val_mean = val_scores.mean(axis=1)

plt.figure()
plt.plot(train_sizes, train_mean, marker='o', label='Treino (F1)')
plt.plot(train_sizes, val_mean, marker='s', label='Validação (F1)')
plt.title(f'Curva de Aprendizado — {best_name}')
plt.xlabel('Tamanho do treino'); plt.ylabel('F1'); plt.legend(); plt.show()



# Amostra para explicações
sample_size = min(150, len(X_val))
X_val_sample = X_val.sample(sample_size, random_state=RANDOM_STATE)

HAS_SHAP_VALUES = False
if HAS_SHAP:
    try:
        # Função de previsão contínua
        if hasattr(best_estimator, "predict_proba"):
            f = lambda data: best_estimator.predict_proba(pd.DataFrame(data, columns=X.columns))[:,1]
        elif hasattr(best_estimator, "decision_function"):
            def f(data):
                scores = best_estimator.decision_function(pd.DataFrame(data, columns=X.columns))
                smin, smax = scores.min(), scores.max()
                return ((scores - smin) / (smax - smin + 1e-9)).reshape(-1,)
        else:
            f = lambda data: best_estimator.predict(pd.DataFrame(data, columns=X.columns)).astype(float)

        background = X_train.sample(min(200, len(X_train)), random_state=RANDOM_STATE)
        explainer = shap.KernelExplainer(f, background, link="identity")
        shap_values = explainer.shap_values(X_val_sample, nsamples=200)

        shap_abs_mean = np.abs(shap_values).mean(axis=0)
        imp = pd.DataFrame({'feature': X.columns, 'mean_abs_shap': shap_abs_mean}).sort_values('mean_abs_shap', ascending=False)
        display(imp.head(20))

        topn = 15
        top_imp = imp.head(topn).iloc[::-1]
        plt.figure()
        plt.barh(top_imp['feature'], top_imp['mean_abs_shap'])
        plt.title('Importância (|SHAP| médio) — Top Features')
        plt.xlabel('|SHAP| médio'); plt.ylabel('Feature'); plt.tight_layout(); plt.show()
        HAS_SHAP_VALUES = True
    except Exception as e:
        print("Falha SHAP, usando permutation importance. Erro:", e)

if not HAS_SHAP_VALUES:
    print("SHAP indisponível; seguindo com permutation importance.")
    
# Permutation importance (sempre mostramos)
try:
    r = permutation_importance(best_estimator, X_val, y_val, n_repeats=10, random_state=RANDOM_STATE, scoring='f1')
    imp_perm = pd.DataFrame({'feature': X.columns, 'importance_mean': r.importances_mean, 'importance_std': r.importances_std})
    imp_perm = imp_perm.sort_values('importance_mean', ascending=False)
    display(imp_perm.head(20))

    plt.figure()
    topn = 15
    plot_df = imp_perm.head(topn).iloc[::-1]
    plt.barh(plot_df['feature'], plot_df['importance_mean'])
    plt.title('Permutation Importance (F1) — Top Features')
    plt.xlabel('Importância média'); plt.ylabel('Feature'); plt.tight_layout(); plt.show()
except Exception as e:
    print("Falha em permutation importance:", e)



# Pontuações no conjunto de validação
if hasattr(best_estimator, "predict_proba"):
    scores_val = best_estimator.predict_proba(X_val)[:,1]
elif hasattr(best_estimator, "decision_function"):
    s = best_estimator.decision_function(X_val)
    smin, smax = s.min(), s.max()
    scores_val = (s - smin) / (smax - smin + 1e-9)
else:
    scores_val = best_estimator.predict(X_val).astype(float)

thresholds = np.linspace(0.0, 1.0, 201)
rows = []
for t in thresholds:
    y_pred_t = (scores_val >= t).astype(int)
    prec = precision_score(y_val, y_pred_t, zero_division=0)
    rec = recall_score(y_val, y_pred_t, zero_division=0)
    rows.append({
        'threshold': t,
        'accuracy': accuracy_score(y_val, y_pred_t),
        'precision': prec,
        'recall': rec,
        'f1': f1_score(y_val, y_pred_t, zero_division=0),
        'f2': (5*prec*rec) / (4*prec + rec + 1e-9)
    })
thr_df = pd.DataFrame(rows)

TARGET_RECALL = 0.98  # ajuste conforme necessidade
best_f2 = thr_df.iloc[thr_df['f2'].idxmax()]
candidates = thr_df[thr_df['recall'] >= TARGET_RECALL]
best_recall_thr = candidates.iloc[candidates['precision'].idxmax()] if len(candidates) else None

display(thr_df.head())
print("Melhor por F2:", best_f2.to_dict())
if best_recall_thr is not None:
    print(f"Melhor recall >= {TARGET_RECALL}: ", best_recall_thr.to_dict())

plt.figure()
plt.plot(thresholds, thr_df['precision'], label='Precision')
plt.plot(thresholds, thr_df['recall'], label='Recall')
plt.plot(thresholds, thr_df['f1'], label='F1')
plt.plot(thresholds, thr_df['f2'], label='F2')
plt.title('Métricas vs limiar (validação)')
plt.xlabel('Limiar'); plt.ylabel('Score'); plt.legend(); plt.show()

final_threshold = float(best_f2['threshold'])
print("Limiar final escolhido:", final_threshold)



# Curva de calibração (validação)
if hasattr(best_estimator, "predict_proba"):
    prob_pos = best_estimator.predict_proba(X_val)[:,1]
elif hasattr(best_estimator, "decision_function"):
    s = best_estimator.decision_function(X_val)
    smin, smax = s.min(), s.max()
    prob_pos = (s - smin) / (smax - smin + 1e-9)
else:
    prob_pos = best_estimator.predict(X_val).astype(float)

frac_pos, mean_pred = calibration_curve(y_val, prob_pos, n_bins=10, strategy='quantile')

plt.figure()
plt.plot([0,1], [0,1], linestyle='--')
plt.plot(mean_pred, frac_pos, marker='o')
plt.title('Curva de Calibração — Validação')
plt.xlabel('Prob. prevista (média no bin)'); plt.ylabel('Fração positiva observada'); plt.show()

from sklearn.metrics import brier_score_loss
print("Brier score (validação):", brier_score_loss(y_val, prob_pos))



# Avaliar no TESTE com limiar escolhido
if hasattr(best_estimator, "predict_proba"):
    test_scores = best_estimator.predict_proba(X_test)[:,1]
elif hasattr(best_estimator, "decision_function"):
    s = best_estimator.decision_function(X_test)
    smin, smax = s.min(), s.max()
    test_scores = (s - smin) / (smax - smin + 1e-9)
else:
    test_scores = best_estimator.predict(X_test).astype(float)

y_test_pred_thr = (test_scores >= final_threshold).astype(int)

print("== TESTE (limiar customizado) ==")
print("Accuracy:", accuracy_score(y_test, y_test_pred_thr))
print("Precision:", precision_score(y_test, y_test_pred_thr, zero_division=0))
print("Recall:", recall_score(y_test, y_test_pred_thr, zero_division=0))
print("F1:", f1_score(y_test, y_test_pred_thr, zero_division=0))
print("\nClassification Report (teste):\n", classification_report(y_test, y_test_pred_thr, digits=4))

cm = confusion_matrix(y_test, y_test_pred_thr)
fig, ax = plt.subplots()
im = ax.imshow(cm)
ax.set_title(f'Matriz de Confusão — Teste (limiar {final_threshold:.2f})')
ax.set_xlabel('Predito'); ax.set_ylabel('Verdadeiro')
ax.set_xticks([0,1]); ax.set_yticks([0,1])
ax.set_xticklabels(['Benigno (0)', 'Maligno (1)']); ax.set_yticklabels(['Benigno (0)', 'Maligno (1)'])
for (i, j), z in np.ndenumerate(cm):
    ax.text(j, i, str(z), ha='center', va='center')
fig.colorbar(im); plt.show()

# Persistir modelo, features e limiar
bundle_path = 'best_model_with_threshold.joblib'
joblib.dump({'model': best_estimator, 'features': list(X.columns), 'threshold': float(final_threshold)}, bundle_path)
print(f"Bundle salvo em: {bundle_path}")
