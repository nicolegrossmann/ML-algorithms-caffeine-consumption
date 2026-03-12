from __future__ import annotations

import json
from pathlib import Path

from .evaluation import compare_unsupervised_supervised, format_model_metrics
from .preprocessing import preprocess, read_tables
from .supervised import run_supervised
from .unsupervised import run_unsupervised


def run_full_pipeline(expression_path: str, metadata_path: str, output_dir: str = "reports") -> dict[str, object]:
    expr, meta = read_tables(expression_path, metadata_path)
    x, y, clean_meta, genes = preprocess(expr, meta)

    unsup = run_unsupervised(x, y)
    sup = run_supervised(x, y, metadata=clean_meta)
    comparison = compare_unsupervised_supervised(unsup, sup)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    metrics_df = format_model_metrics(sup)
    metrics_df.to_csv(out / "supervised_metrics.csv", index=False)
    sup["top_logreg_features"].to_csv(out / "top_logreg_features.csv")
    sup["top_rf_features"].to_csv(out / "top_rf_features.csv")

    with (out / "comparison_summary.json").open("w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)

    with (out / "pipeline_overview.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "n_samples": int(x.shape[0]),
                "n_genes_after_filtering": int(len(genes)),
                "labels": sorted(list(set(y.values))),
            },
            f,
            indent=2,
        )

    return {
        "unsupervised": unsup,
        "supervised": sup,
        "comparison": comparison,
    }

