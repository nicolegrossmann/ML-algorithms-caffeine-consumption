from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


REQUIRED_METADATA_COLUMNS = {"sample_id", "condition"}


@dataclass
class PreprocessingConfig:
    min_gene_count: float = 10.0
    min_samples_fraction: float = 0.2
    log_transform: bool = True
    pseudocount: float = 1.0
    standardize: bool = True


def read_tables(expression_path: str, metadata_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    expression = pd.read_csv(expression_path)
    metadata = pd.read_csv(metadata_path)
    return expression, metadata


def validate_schema(expression: pd.DataFrame, metadata: pd.DataFrame) -> None:
    if "sample_id" not in expression.columns:
        raise ValueError("Expression table must include a 'sample_id' column.")

    missing = REQUIRED_METADATA_COLUMNS.difference(metadata.columns)
    if missing:
        raise ValueError(f"Metadata table missing required columns: {sorted(missing)}")

    expr_ids = set(expression["sample_id"])
    meta_ids = set(metadata["sample_id"])
    if expr_ids != meta_ids:
        diff_expr = sorted(expr_ids - meta_ids)[:5]
        diff_meta = sorted(meta_ids - expr_ids)[:5]
        raise ValueError(
            "Sample IDs do not match between expression and metadata. "
            f"Only in expression (first 5): {diff_expr}; only in metadata (first 5): {diff_meta}"
        )


def merge_expression_metadata(
    expression: pd.DataFrame,
    metadata: pd.DataFrame,
) -> pd.DataFrame:
    validate_schema(expression, metadata)
    merged = metadata.merge(expression, on="sample_id", how="inner")
    return merged


def get_gene_columns(df: pd.DataFrame, exclude: Optional[Iterable[str]] = None) -> list[str]:
    exclude_set = set(exclude or [])
    exclude_set.update({"sample_id", "condition", "subject_id", "timepoint", "nutrition_group", "response_label"})
    return [col for col in df.columns if col not in exclude_set]


def filter_low_expression_genes(
    merged: pd.DataFrame,
    gene_columns: list[str],
    min_gene_count: float,
    min_samples_fraction: float,
) -> list[str]:
    min_samples = max(1, int(np.ceil(min_samples_fraction * merged.shape[0])))
    keep_mask = (merged[gene_columns] >= min_gene_count).sum(axis=0) >= min_samples
    filtered_genes = list(pd.Index(gene_columns)[keep_mask.values])
    if not filtered_genes:
        raise ValueError("No genes remain after filtering. Lower thresholds in PreprocessingConfig.")
    return filtered_genes


def build_feature_matrix(
    merged: pd.DataFrame,
    gene_columns: list[str],
    config: PreprocessingConfig,
) -> pd.DataFrame:
    x = merged[gene_columns].copy()

    if config.log_transform:
        x = np.log2(x + config.pseudocount)

    if config.standardize:
        scaler = StandardScaler()
        x = pd.DataFrame(scaler.fit_transform(x), columns=gene_columns, index=merged.index)

    return x


def preprocess(
    expression: pd.DataFrame,
    metadata: pd.DataFrame,
    config: Optional[PreprocessingConfig] = None,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, list[str]]:
    cfg = config or PreprocessingConfig()
    merged = merge_expression_metadata(expression, metadata)
    genes = get_gene_columns(merged)
    filtered_genes = filter_low_expression_genes(
        merged=merged,
        gene_columns=genes,
        min_gene_count=cfg.min_gene_count,
        min_samples_fraction=cfg.min_samples_fraction,
    )

    x = build_feature_matrix(merged, filtered_genes, cfg)
    y = merged["condition"].copy()
    meta = merged[[c for c in merged.columns if c not in filtered_genes]].copy()
    return x, y, meta, filtered_genes

