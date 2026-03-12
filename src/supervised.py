from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler


@dataclass
class SupervisedConfig:
    n_splits: int = 5
    random_state: int = 42


def _safe_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    try:
        return float(roc_auc_score(y_true, y_score))
    except ValueError:
        return float("nan")


def _evaluate_binary(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
        "roc_auc": _safe_roc_auc(y_true, y_score),
    }


def _build_cv(y_encoded: np.ndarray, groups: Optional[pd.Series], n_splits: int, random_state: int):
    if groups is not None:
        unique_groups = np.unique(groups.values)
        splits = min(n_splits, unique_groups.shape[0])
        if splits < 2:
            raise ValueError("Need at least 2 unique groups in 'subject_id' for group-aware CV.")
        return GroupKFold(n_splits=splits)

    class_counts = np.bincount(y_encoded)
    max_possible = int(np.min(class_counts)) if class_counts.size else 2
    splits = max(2, min(n_splits, max_possible))
    return StratifiedKFold(n_splits=splits, shuffle=True, random_state=random_state)


def run_supervised(
    x: pd.DataFrame,
    y: pd.Series,
    metadata: Optional[pd.DataFrame] = None,
    config: Optional[SupervisedConfig] = None,
) -> dict[str, object]:
    cfg = config or SupervisedConfig()

    le = LabelEncoder()
    y_encoded = le.fit_transform(y.values)
    if len(le.classes_) != 2:
        raise ValueError("Current implementation expects a binary target label.")

    groups = None
    if metadata is not None and "subject_id" in metadata.columns:
        subject_col = metadata["subject_id"]
        if subject_col.notna().all():
            groups = subject_col

    cv = _build_cv(y_encoded, groups, cfg.n_splits, cfg.random_state)
    cv_kwargs = {"cv": cv, "method": "predict"}
    if groups is not None:
        cv_kwargs["groups"] = groups

    lr = Pipeline(
        steps=[
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=cfg.random_state)),
        ]
    )

    rf = RandomForestClassifier(
        n_estimators=300,
        random_state=cfg.random_state,
        class_weight="balanced",
    )

    y_pred_lr = cross_val_predict(lr, x, y_encoded, **cv_kwargs)
    y_pred_rf = cross_val_predict(rf, x, y_encoded, **cv_kwargs)

    proba_kwargs = {"cv": cv, "method": "predict_proba"}
    if groups is not None:
        proba_kwargs["groups"] = groups
    y_score_lr = cross_val_predict(lr, x, y_encoded, **proba_kwargs)[:, 1]
    y_score_rf = cross_val_predict(rf, x, y_encoded, **proba_kwargs)[:, 1]

    lr_metrics = _evaluate_binary(y_encoded, y_pred_lr, y_score_lr)
    rf_metrics = _evaluate_binary(y_encoded, y_pred_rf, y_score_rf)
    lr_cm = confusion_matrix(y_encoded, y_pred_lr).tolist()
    rf_cm = confusion_matrix(y_encoded, y_pred_rf).tolist()

    lr.fit(x, y_encoded)
    rf.fit(x, y_encoded)
    lr_coef = lr.named_steps["clf"].coef_[0]
    rf_importance = rf.feature_importances_

    top_lr = (
        pd.Series(lr_coef, index=x.columns)
        .sort_values(key=lambda s: s.abs(), ascending=False)
        .head(20)
        .rename("logreg_weight")
    )
    top_rf = pd.Series(rf_importance, index=x.columns).sort_values(ascending=False).head(20).rename("rf_importance")

    return {
        "label_classes": list(le.classes_),
        "logistic_regression_metrics": lr_metrics,
        "random_forest_metrics": rf_metrics,
        "logistic_regression_confusion_matrix": lr_cm,
        "random_forest_confusion_matrix": rf_cm,
        "top_logreg_features": top_lr,
        "top_rf_features": top_rf,
    }

