"""Spherical k-means baseline."""

from __future__ import annotations

import warnings

import numpy as np
from scipy import sparse
from sklearn.cluster import KMeans
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
