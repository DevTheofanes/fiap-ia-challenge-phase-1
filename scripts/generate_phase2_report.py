from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
import sys

sys.path.insert(0, str(BASE_DIR))

from src import config
from src.data.load import load_dataset
from src.data.split import apply_splits, make_or_load_splits
from src.models.train import evaluate_on_test


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gerar relatorio da Fase 2 (GA vs Baseline).")
    parser.add_argument("--model", choices=["LR", "RF", "all"], default="all")
    parser.add_argument("--runs-root", type=Path, default=config.ARTIFACTS_DIR / "ga_runs")
    parser.add_argument("--summary-root", type=Path, default=config.ARTIFACTS_DIR / "ga_summary")
    parser.add_argument("--reports-root", type=Path, default=BASE_DIR / "reports")
    return parser.parse_args()


def _load_data_splits():
    X, y = load_dataset(config.DATA_PATH, config.TARGET_COL, config.ID_COLS)
    splits = make_or_load_splits(
        n_samples=len(X),
        y=y,
        test_size=config.TEST_SIZE,
        val_size=config.VAL_SIZE,
        random_state=config.RANDOM_STATE,
        splits_path=config.SPLITS_PATH,
    )
    return apply_splits(X, y, splits)


def _load_baseline_models(models_dir: Path):
    models = {}
    for model_key in ["LR", "RF"]:
        model_path = models_dir / f"{model_key}.joblib"
        if model_path.exists():
            models[model_key] = joblib.load(model_path)
    return models


def _compute_baseline_metrics(models_dir: Path, data_splits):
    X_train, X_val, X_test, y_train, y_val, y_test = data_splits
    models = _load_baseline_models(models_dir)
    metrics = {}
    for model_key, model in models.items():
        metrics[model_key] = {
            "val": evaluate_on_test(model, X_val, y_val),
            "test": evaluate_on_test(model, X_test, y_test),
        }
    return metrics


def _load_baseline_params(metrics_path: Path):
    if not metrics_path.exists():
        return {}
    try:
        payload = json.loads(metrics_path.read_text())
    except Exception:
        return {}
    params = {}
    for model_key, model_payload in payload.get("models", {}).items():
        params[model_key] = model_payload.get("params", {})
    return params


def _collect_ga_runs(runs_root: Path, model: str, data_splits):
    runs = []
    model_dir = runs_root / model
    if not model_dir.exists():
        return runs
    X_train, X_val, X_test, y_train, y_val, y_test = data_splits

    for best_path in model_dir.glob("exp*_seed*/best.json"):
        try:
            payload = json.loads(best_path.read_text())
        except Exception:
            continue
        holdout = payload.get("holdout") or "val"
        model_path = best_path.parent / "best_model.joblib"
        if not model_path.exists():
            continue
        ga_model = joblib.load(model_path)
        if holdout == "val":
            holdout_metrics = evaluate_on_test(ga_model, X_val, y_val)
        else:
            holdout_metrics = evaluate_on_test(ga_model, X_test, y_test)
        config_payload = payload.get("config", {})
        runs.append(
            {
                "model": model,
                "exp_id": config_payload.get("exp_id"),
                "seed": config_payload.get("seed"),
                "best_cv_f1": payload.get("best_cv_f1"),
                "holdout_split": holdout,
                "holdout_f1": holdout_metrics.get("f1"),
                "holdout_accuracy": holdout_metrics.get("accuracy"),
                "holdout_precision": holdout_metrics.get("precision"),
                "holdout_recall": holdout_metrics.get("recall"),
                "holdout_roc_auc": holdout_metrics.get("roc_auc"),
                "holdout_pr_auc": holdout_metrics.get("pr_auc"),
                "training_time_sec": payload.get("training_time_sec"),
                "params": payload.get("decoded_params", {}),
                "ga_config": config_payload.get("ga_config", {}),
                "run_dir": str(best_path.parent),
            }
        )
    return runs


def _format_metric(value):
    if value is None:
        return ""
    try:
        return f"{float(value):.4f}"
    except Exception:
        return ""


def _format_mean_std(mean_val, std_val):
    if mean_val is None:
        return ""
    mean_str = f"{float(mean_val):.4f}"
    if std_val is None or pd.isna(std_val):
        return mean_str
    return f"{mean_str} +/- {float(std_val):.4f}"


def _params_summary(params: dict, model_key: str) -> str:
    if not isinstance(params, dict) or not params:
        return "-"
    preferred = {
        "LR": ["C", "penalty", "solver", "class_weight", "max_iter"],
        "RF": ["n_estimators", "max_depth", "min_samples_split", "min_samples_leaf", "max_features"],
    }
    keys = preferred.get(model_key, list(params.keys()))
    parts = []
    for key in keys:
        if key not in params:
            continue
        value = params.get(key)
        parts.append(f"{key}={value}")
        if len(parts) >= 4:
            break
    return ", ".join(parts) if parts else "-"


def _plot_convergence(runs_root: Path, model: str, figures_dir: Path):
    rows = []
    model_dir = runs_root / model
    if not model_dir.exists():
        return None
    for history_path in model_dir.glob("exp*_seed*/history.csv"):
        df = pd.read_csv(history_path)
        rows.append(df)
    if not rows:
        return None
    history = pd.concat(rows, ignore_index=True)
    grouped = history.groupby(["exp_id", "generation"], as_index=False)["best_f1"].mean()
    plt.figure(figsize=(7, 4))
    for exp_id in sorted(grouped["exp_id"].unique()):
        exp_df = grouped[grouped["exp_id"] == exp_id]
        plt.plot(exp_df["generation"], exp_df["best_f1"], label=f"Exp {exp_id}")
    plt.xlabel("Geracao")
    plt.ylabel("Melhor F1 (media)")
    plt.title(f"Convergencia do GA - {model}")
    plt.legend()
    figures_dir.mkdir(parents=True, exist_ok=True)
    out_path = figures_dir / f"ga_convergence_{model}.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()
    return out_path


def _plot_baseline_vs_ga(model: str, baseline_metrics: dict, ga_metrics: dict, figures_dir: Path):
    labels = ["F1", "Recall"]
    baseline_vals = [baseline_metrics.get("f1"), baseline_metrics.get("recall")]
    ga_vals = [ga_metrics.get("f1"), ga_metrics.get("recall")]

    x = range(len(labels))
    width = 0.35
    plt.figure(figsize=(6, 4))
    plt.bar([i - width / 2 for i in x], baseline_vals, width, label="Baseline")
    plt.bar([i + width / 2 for i in x], ga_vals, width, label="GA")
    plt.xticks(list(x), labels)
    plt.ylim(0.0, 1.0)
    plt.ylabel("Score")
    plt.title(f"Baseline vs GA - {model}")
    plt.legend()
    figures_dir.mkdir(parents=True, exist_ok=True)
    out_path = figures_dir / f"baseline_vs_ga_{model}.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()
    return out_path


def _plot_precision_recall(model: str, baseline_metrics: dict, ga_metrics: dict, figures_dir: Path):
    plt.figure(figsize=(6, 4))
    plt.scatter(baseline_metrics.get("recall"), baseline_metrics.get("precision"), label="Baseline", s=80)
    plt.scatter(ga_metrics.get("recall"), ga_metrics.get("precision"), label="GA", s=80)
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.xlabel("Recall")
    plt.ylabel("Precisao")
    plt.title(f"Precisao vs Recall - {model}")
    plt.legend()
    figures_dir.mkdir(parents=True, exist_ok=True)
    out_path = figures_dir / f"precision_vs_recall_{model}.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()
    return out_path


def main() -> None:
    args = _parse_args()
    models = ["LR", "RF"] if args.model == "all" else [args.model]

    data_splits = _load_data_splits()
    baseline_metrics = _compute_baseline_metrics(config.BASELINE_MODELS_DIR, data_splits)
    baseline_params = _load_baseline_params(config.BASELINE_METRICS_PATH)

    all_runs = []
    for model in models:
        all_runs.extend(_collect_ga_runs(args.runs_root, model, data_splits))
    if not all_runs:
        print("Nenhuma execucao do GA encontrada.")
        return

    runs_df = pd.DataFrame(all_runs)
    best_runs = {}
    for model in models:
        model_runs = runs_df[runs_df["model"] == model]
        if model_runs.empty:
            continue
        best_row = model_runs.sort_values(
            ["holdout_f1", "best_cv_f1"], ascending=False
        ).iloc[0]
        best_runs[model] = best_row.to_dict()

    stability_rows = []
    grouped = runs_df.groupby(["model", "exp_id"])
    for (model, exp_id), df in grouped:
        stability_rows.append(
            {
                "model": model,
                "exp_id": exp_id,
                "seeds": df["seed"].nunique(),
                "best_cv_f1_mean": df["best_cv_f1"].mean(),
                "best_cv_f1_std": df["best_cv_f1"].std(),
                "holdout_f1_mean": df["holdout_f1"].mean(),
                "holdout_f1_std": df["holdout_f1"].std(),
                "holdout_recall_mean": df["holdout_recall"].mean(),
                "holdout_recall_std": df["holdout_recall"].std(),
                "training_time_mean": df["training_time_sec"].mean(),
            }
        )
    stability_df = pd.DataFrame(stability_rows)
    args.summary_root.mkdir(parents=True, exist_ok=True)
    stability_df.to_csv(args.summary_root / "ga_stability_summary.csv", index=False)

    figures_dir = args.reports_root / "figures"
    figure_paths = []
    for model in models:
        figure_paths.append(_plot_convergence(args.runs_root, model, figures_dir))
        if model not in best_runs:
            continue
        holdout = best_runs[model].get("holdout_split") or "val"
        baseline = baseline_metrics.get(model, {}).get(holdout, {})
        ga_metrics = {
            "f1": best_runs[model].get("holdout_f1"),
            "recall": best_runs[model].get("holdout_recall"),
            "precision": best_runs[model].get("holdout_precision"),
        }
        figure_paths.append(_plot_baseline_vs_ga(model, baseline, ga_metrics, figures_dir))
        figure_paths.append(_plot_precision_recall(model, baseline, ga_metrics, figures_dir))

    report_lines = []
    report_lines.append("# Fase 2 - Resultados GA vs Baseline")
    report_lines.append("")
    report_lines.append("## Tabela 1 - Baseline vs GA (melhor execucao)")
    report_lines.append(
        "| Modelo | Abordagem | Params | F1 (CV) | F1 (Holdout) | Recall | Precisao | ROC-AUC | PR-AUC |"
    )
    report_lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for model in models:
        if model not in best_runs:
            continue
        holdout = best_runs[model].get("holdout_split") or "val"
        baseline = baseline_metrics.get(model, {}).get(holdout, {})
        report_lines.append(
            "| {model} | Baseline | {params} |  | {f1} | {recall} | {precision} | {roc} | {pr} |".format(
                model=model,
                params=_params_summary(baseline_params.get(model, {}), model),
                f1=_format_metric(baseline.get("f1")),
                recall=_format_metric(baseline.get("recall")),
                precision=_format_metric(baseline.get("precision")),
                roc=_format_metric(baseline.get("roc_auc")),
                pr=_format_metric(baseline.get("pr_auc")),
            )
        )
        report_lines.append(
            "| {model} | GA | {params} | {f1_cv} | {f1} | {recall} | {precision} | {roc} | {pr} |".format(
                model=model,
                params=_params_summary(best_runs[model].get("params", {}), model),
                f1_cv=_format_metric(best_runs[model].get("best_cv_f1")),
                f1=_format_metric(best_runs[model].get("holdout_f1")),
                recall=_format_metric(best_runs[model].get("holdout_recall")),
                precision=_format_metric(best_runs[model].get("holdout_precision")),
                roc=_format_metric(best_runs[model].get("holdout_roc_auc")),
                pr=_format_metric(best_runs[model].get("holdout_pr_auc")),
            )
        )

    report_lines.append("")
    report_lines.append("## Tabela 2 - Estabilidade do GA por experimento")
    report_lines.append(
        "| Modelo | Exp | Seeds | Melhor F1 CV (media +/- desvio) | F1 Holdout (media +/- desvio) | Recall Holdout (media +/- desvio) | Tempo medio (s) |"
    )
    report_lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for _, row in stability_df.iterrows():
        report_lines.append(
            "| {model} | {exp} | {seeds} | {cv} | {f1} | {recall} | {time} |".format(
                model=row["model"],
                exp=row["exp_id"],
                seeds=int(row["seeds"]),
                cv=_format_mean_std(row["best_cv_f1_mean"], row["best_cv_f1_std"]),
                f1=_format_mean_std(row["holdout_f1_mean"], row["holdout_f1_std"]),
                recall=_format_mean_std(row["holdout_recall_mean"], row["holdout_recall_std"]),
                time=f"{row['training_time_mean']:.1f}" if pd.notna(row["training_time_mean"]) else "",
            )
        )

    report_lines.append("")
    report_lines.append("## Figuras")
    for path in figure_paths:
        if path:
            rel_path = path.relative_to(args.reports_root)
            report_lines.append(f"![{path.stem}]({rel_path.as_posix()})")

    report_lines.append("")
    report_lines.append("## Discussao critica")
    for model in models:
        if model not in best_runs:
            continue
        holdout = best_runs[model].get("holdout_split") or "val"
        baseline = baseline_metrics.get(model, {}).get(holdout, {})
        ga = best_runs[model]
        base_f1 = baseline.get("f1")
        ga_f1 = ga.get("holdout_f1")
        base_recall = baseline.get("recall")
        ga_recall = ga.get("holdout_recall")
        base_precision = baseline.get("precision")
        ga_precision = ga.get("holdout_precision")
        if base_f1 is not None and ga_f1 is not None and base_f1 > 0:
            delta = ga_f1 - base_f1
            delta_pct = delta / base_f1 * 100.0
            report_lines.append(
                f"- {model}: GA melhora o F1 no holdout de {base_f1:.4f} para {ga_f1:.4f} (delta {delta:.4f}, {delta_pct:.2f}%)."
            )
        if base_recall is not None and ga_recall is not None:
            report_lines.append(
                f"- {model}: Recall muda de {base_recall:.4f} para {ga_recall:.4f}, impactando falsos negativos."
            )
        if base_precision is not None and ga_precision is not None:
            report_lines.append(
                f"- {model}: Precisao muda de {base_precision:.4f} para {ga_precision:.4f}, afetando falsos positivos."
            )
        gap = None
        if ga.get("best_cv_f1") is not None and ga_f1 is not None:
            gap = ga.get("best_cv_f1") - ga_f1
        if gap is not None:
            report_lines.append(
                f"- {model}: Diferenca entre CV e holdout de {gap:.4f}; gaps altos sugerem overfitting."
            )
        model_stability = stability_df[stability_df["model"] == model]
        if not model_stability.empty:
            best_exp = model_stability.sort_values("holdout_f1_std").iloc[0]
            report_lines.append(
                f"- {model}: Exp {best_exp['exp_id']} mais estavel (menor desvio do F1 holdout)."
            )
        ga_config = ga.get("ga_config", {})
        if ga_config:
            pop = ga_config.get("pop_size")
            gens = ga_config.get("n_generations")
            if pop and gens:
                evals = pop * gens * 3
                report_lines.append(
                    f"- {model}: Custo aprox {evals} avaliacoes de CV (pop {pop} x gen {gens} x 3 folds)."
                )

    report_lines.append("- Limitacoes: dataset pequeno e espaco de busca limitado; resultados sensiveis ao split.")
    report_lines.append("- Proximos passos: ampliar espaco de busca, adicionar calibracao e considerar early stopping.")

    args.reports_root.mkdir(parents=True, exist_ok=True)
    report_path = args.reports_root / "phase2_results.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"Relatorio salvo em {report_path}")


if __name__ == "__main__":
    main()
