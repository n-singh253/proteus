"""Spherical k-means baseline."""

from __future__ import annotations

import warnings

import numpy as np
from scipy import sparse
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import normalize


class CosineKMeans:
    """Lloyd k-means on unit vectors with re-normalized centroids.

    On the unit sphere squared Euclidean distance is 2 - 2*cos, so this is a close, well-tested
    approximation of spherical k-means. Fewer than ``k`` prototypes are returned when the data
    has fewer distinct points than requested clusters.
    """

    def __init__(self, n_init: int = 4, max_iter: int = 300) -> None:
        self.n_init = n_init
        self.max_iter = max_iter

    def prototypes(self, features: sparse.csr_matrix, k: int, seed: int) -> np.ndarray:
        k = min(k, features.shape[0])
        if k == 0:
            return np.zeros((0, features.shape[1]))
        model = KMeans(n_clusters=k, n_init=self.n_init, max_iter=self.max_iter, random_state=seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            model.fit(features)
        centers = normalize(model.cluster_centers_)
        return np.unique(np.round(centers, 9), axis=0)


class MergedKMeans:
    """Over-cluster with ``CosineKMeans``, then merge micro-centroids by average linkage.

    Plain k-means minimizes size-weighted error, so it prefers splitting a large group over
    isolating a small distinct one. Sub-clusters of a large group sit close together and
    merge back, while small distinct groups stay separate. Merging works on at most a few
    dozen centroids, so the cost is one larger k-means fit.
    """

    def __init__(self, oversample: float = 2.0, extra: int = 8) -> None:
        self.base = CosineKMeans()
        self.oversample = oversample
        self.extra = extra

    def prototypes(self, features: sparse.csr_matrix, k: int, seed: int) -> np.ndarray:
        micro = self.base.prototypes(features, int(k * self.oversample) + self.extra, seed)
        if len(micro) <= k:
            return micro
        labels = np.asarray(features @ micro.T).argmax(axis=1)
        sizes = np.bincount(labels, minlength=len(micro)).astype(float)
        groups = AgglomerativeClustering(
            n_clusters=k, metric="cosine", linkage="average"
        ).fit_predict(micro)
        merged = np.vstack(
            [(sizes[groups == g, None] * micro[groups == g]).sum(0) for g in range(k)]
        )
        return normalize(merged)
