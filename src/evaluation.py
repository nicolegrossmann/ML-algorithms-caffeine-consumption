from __future__ import annotations

import pandas as pd


def format_model_metrics(result: dict[str, object]) -> pd.DataFrame:
    rows = [
        {
            "model": "logistic_regression",
            **result["logistic_regression_metrics"],
        },
        {
            "model": "random_forest",
            **result["random_forest_metrics"],
        },
    ]
    return pd.DataFrame(rows)


def compare_unsupervised_supervised(
    unsup_result: dict[str, object],
    sup_result: dict[str, object],
) -> dict[str, object]:
    return {
        "unsupervised": {
            "kmeans_silhouette": unsup_result.get("kmeans_silhouette"),
            "hier_silhouette": unsup_result.get("hier_silhouette"),
        },
        "supervised": {
            "logistic_regression_metrics": sup_result.get("logistic_regression_metrics"),
            "random_forest_metrics": sup_result.get("random_forest_metrics"),
        },
    }

