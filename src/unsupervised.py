from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

try:
    import umap
except ImportError:  # pragma: no cover
    umap = None


@dataclass
class UnsupervisedConfig:
    n_pca_components: int = 2
    n_clusters: int = 2
    random_state: int = 42
    umap_n_neighbors: int = 8
    umap_min_dist: float = 0.2


def run_pca(x: pd.DataFrame, n_components: int = 2) -> pd.DataFrame:
    pca = PCA(n_components=n_components)
    coords = pca.fit_transform(x)
    out = pd.DataFrame(coords, index=x.index, columns=[f"PC{i+1}" for i in range(n_components)])
    out.attrs["explained_variance_ratio"] = pca.explained_variance_ratio_
    return out


def run_umap(
    x: pd.DataFrame,
    n_neighbors: int = 8,
    min_dist: float = 0.2,
    random_state: int = 42,
) -> pd.DataFrame:
    if umap is None:
        raise ImportError("umap-learn is not installed. Install dependencies from requirements.txt.")

    safe_neighbors = min(n_neighbors, max(2, x.shape[0] - 1))
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=safe_neighbors,
        min_dist=min_dist,
        random_state=random_state,
    )
    coords = reducer.fit_transform(x)
    return pd.DataFrame(coords, index=x.index, columns=["UMAP1", "UMAP2"])


def cluster_kmeans(x: pd.DataFrame, n_clusters: int = 2, random_state: int = 42) -> pd.Series:
    model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=20)
    labels = model.fit_predict(x)
    return pd.Series(labels, index=x.index, name="kmeans_cluster")


def cluster_hierarchical(x: pd.DataFrame, n_clusters: int = 2) -> pd.Series:
    z = linkage(x.values, method="ward")
    labels = fcluster(z, t=n_clusters, criterion="maxclust") - 1
    return pd.Series(labels, index=x.index, name="hier_cluster")


def compute_silhouette(x: pd.DataFrame, labels: pd.Series) -> float:
    unique_labels = np.unique(labels.values)
    if unique_labels.shape[0] < 2:
        return float("nan")
    return float(silhouette_score(x, labels))


def cluster_label_purity(cluster_labels: pd.Series, true_labels: pd.Series) -> float:
    contingency = pd.crosstab(cluster_labels, true_labels)
    if contingency.empty:
        return float("nan")
    correct = contingency.max(axis=1).sum()
    total = contingency.sum().sum()
    return float(correct / total) if total else float("nan")


def run_unsupervised(
    x: pd.DataFrame,
    y: Optional[pd.Series] = None,
    config: Optional[UnsupervisedConfig] = None,
) -> dict[str, object]:
    cfg = config or UnsupervisedConfig()

    pca_df = run_pca(x, n_components=cfg.n_pca_components)
    umap_df = run_umap(
        x,
        n_neighbors=cfg.umap_n_neighbors,
        min_dist=cfg.umap_min_dist,
        random_state=cfg.random_state,
    )
    kmeans_labels = cluster_kmeans(x, n_clusters=cfg.n_clusters, random_state=cfg.random_state)
    hier_labels = cluster_hierarchical(x, n_clusters=cfg.n_clusters)

    result = {
        "pca": pca_df,
        "umap": umap_df,
        "kmeans_labels": kmeans_labels,
        "hier_labels": hier_labels,
        "kmeans_silhouette": compute_silhouette(x, kmeans_labels),
        "hier_silhouette": compute_silhouette(x, hier_labels),
    }

    if y is not None:
        result["condition"] = y
        result["kmeans_label_purity"] = cluster_label_purity(kmeans_labels, y)
        result["hier_label_purity"] = cluster_label_purity(hier_labels, y)
    return result

