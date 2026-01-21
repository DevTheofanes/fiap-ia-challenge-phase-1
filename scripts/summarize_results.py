from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
import sys

sys.path.insert(0, str(BASE_DIR))

from src import config
from src.llm.summarize import summarize_experiment


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize baseline vs GA results with LLM.")
    parser.add_argument(
        "--metrics-path",
        type=Path,
        default=config.ARTIFACTS_DIR / "ga_summary" / "compare_baseline_vs_ga.csv",
    )
    return parser.parse_args()


def _format_float(value) -> str:
    if value is None or value == "":
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    try:
        return f"{float(value):.4f}"
    except Exception:
        return str(value)


def _df_to_markdown(df: pd.DataFrame) -> str:
    columns = [
        "model",
        "approach",
        "f1_holdout",
        "precision_holdout",
        "recall_holdout",
        "roc_auc_holdout",
        "pr_auc_holdout",
        "notes",
    ]
    columns = [c for c in columns if c in df.columns]
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = [header, separator]
    for _, row in df.iterrows():
        values = [
            _format_float(row.get(col)) if col.endswith("_holdout") else str(row.get(col, ""))
            for col in columns
        ]
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join(rows)


def main() -> None:
    args = _parse_args()
    if not args.metrics_path.exists():
        raise FileNotFoundError(
            f"Compare file not found: {args.metrics_path}. Run summarize_ga_runs.py first."
        )

    df = pd.read_csv(args.metrics_path)
    table_markdown = _df_to_markdown(df)

    log_path = config.ARTIFACTS_DIR / "llm" / "llm_logs.jsonl"
    summary = summarize_experiment(table_markdown, log_path=log_path)

    output_path = config.ARTIFACTS_DIR / "llm" / "experiment_summaries.md"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().isoformat() + "Z"
    content = [
        f"# Experiment Summary ({timestamp})",
        "",
        "## Metrics Table",
        "",
        table_markdown,
        "",
        "## LLM Summary",
        "",
        f"Summary: {summary.summary}",
        "",
        f"Key improvements: {', '.join(summary.key_improvements) or 'n/a'}",
        f"Trade-offs: {', '.join(summary.tradeoffs) or 'n/a'}",
        f"Limitations: {', '.join(summary.limitations) or 'n/a'}",
    ]
    output_path.write_text("\n".join(content), encoding="utf-8")

    print(output_path)


if __name__ == "__main__":
    main()
