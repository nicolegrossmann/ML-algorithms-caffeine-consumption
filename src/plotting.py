from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_embedding(embedding: pd.DataFrame, labels: pd.Series, title: str, x_col: str, y_col: str):
    fig, ax = plt.subplots(figsize=(7, 5))
    plot_df = embedding.copy()
    plot_df["label"] = labels.values
    sns.scatterplot(data=plot_df, x=x_col, y=y_col, hue="label", s=70, ax=ax)
    ax.set_title(title)
    ax.legend(title="label", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    return fig, ax


def plot_top_features(feature_series: pd.Series, title: str, top_n: int = 15):
    top = feature_series.head(top_n).sort_values()
    fig, ax = plt.subplots(figsize=(7, 5))
    top.plot(kind="barh", ax=ax)
    ax.set_title(title)
    ax.set_xlabel(feature_series.name or "importance")
    plt.tight_layout()
    return fig, ax

