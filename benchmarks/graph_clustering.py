"""Experimental clusterer: spectral clustering of a sparse k-nearest-neighbour cosine graph.

Not part of the package. ``register()`` adds it to the engine's clusterer registry as
``knn-graph`` so benchmarks can run the real pipeline with it.

Each track links to its ``neighbors`` most cosine-similar tracks (on a truncated SVD of the
features for speed; the graph itself is sparse, n x neighbors edges). The symmetrized graph is
embedded with its first k Laplacian eigenvectors, the embedding is k-means clustered, and
cluster means in the original feature space become prototypes, so assignment and every
downstream constraint are unchanged.
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy import sparse
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import spectral_embedding
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

from proteus_music.engine.clustering.registry import CLUSTERERS

NAME = "knn-graph"
LOBPCG_ABOVE = 10_000  # ARPACK's shift-invert factorization measured 1.7 GB at 50k tracks.


class KnnGraphClusterer:
    def __init__(self, neighbors: int = 15, svd_dims: int = 64) -> None:
        self.neighbors, self.svd_dims = neighbors, svd_dims
        self._graph: tuple[tuple, sparse.csr_matrix] | None = None

    def graph(self, features: sparse.csr_matrix, seed: int) -> sparse.csr_matrix:
        key = (id(features), features.shape, features.nnz)
        if self._graph is not None and self._graph[0] == key:
            return self._graph[1]  # Count selection re-clusters the same sample many times.
        dims = min(self.svd_dims, features.shape[1] - 1, features.shape[0] - 1)
        reduced = features
        if dims >= 2 and features.shape[1] > self.svd_dims:
            reduced = TruncatedSVD(dims, random_state=seed).fit_transform(features)
        reduced = normalize(reduced)
        k = min(self.neighbors + 1, features.shape[0])
        distances, indices = NearestNeighbors(n_neighbors=k).fit(reduced).kneighbors(reduced)
        weights = np.clip(1 - distances**2 / 2, 1e-6, None)  # Unit vectors: cos = 1 - d^2/2.
        n = features.shape[0]
        rows = np.repeat(np.arange(n), k)
        graph = sparse.csr_matrix((weights.ravel(), (rows, indices.ravel())), shape=(n, n))
        graph = graph.maximum(graph.T)
        graph.setdiag(0)
        graph.eliminate_zeros()
        self._graph = (key, graph)
        return graph

    def prototypes(self, features: sparse.csr_matrix, k: int, seed: int) -> np.ndarray:
        n = features.shape[0]
        k = min(k, n)
        if k == 0:
            return np.zeros((0, features.shape[1]))
        if k == 1 or n <= self.neighbors + 1:
            return normalize(np.asarray(features.mean(axis=0)))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)  # Disconnected-graph notice.
            embedding = spectral_embedding(
                self.graph(features, seed),
                n_components=k,
                eigen_solver="lobpcg" if n > LOBPCG_ABOVE else "arpack",
                random_state=seed,
                drop_first=False,
            )
        labels = KMeans(k, n_init=4, random_state=seed).fit_predict(normalize(embedding))
        membership = sparse.csr_matrix((np.ones(n), (labels, np.arange(n))), shape=(k, n))
        centers = normalize((membership @ features).toarray())
        return np.unique(np.round(centers[np.bincount(labels, minlength=k) > 0], 9), axis=0)


def register() -> None:
    CLUSTERERS[NAME] = KnnGraphClusterer()
