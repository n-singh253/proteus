"""Prototypes for a mix of user-defined categories and discovered groups.

Discovered prototypes are initialized by the configured clusterer on the residual tracks that
match no category well, so categories do not swallow every loosely related song. A joint
Lloyd refinement then reassigns every track by its overall best prototype. Category
prototypes stay anchored to their query vector (half weight) so they keep their meaning while
absorbing the artist and era profile of their members, which makes their similarities
comparable with discovered prototypes.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse
from sklearn.preprocessing import normalize

from proteus_music.engine.clustering.registry import Clusterer

RESIDUAL_THRESHOLD = 0.2
ANCHOR_WEIGHT = 0.5
MAX_REFINE_ITERATIONS = 25


def build_prototypes(
    features: sparse.csr_matrix,
    anchors: np.ndarray,
    n_discovered: int,
    clusterer: Clusterer,
    seed: int,
) -> np.ndarray:
    """Rows: one per anchor (in order), then up to ``n_discovered`` discovered prototypes."""
    if len(anchors) == 0:
        return clusterer.prototypes(features, n_discovered, seed)
    free = np.zeros((0, features.shape[1]))
    if n_discovered > 0:
        best = np.asarray(features @ anchors.T).max(axis=1)
        residual = features[best < RESIDUAL_THRESHOLD]
        pool = residual if residual.shape[0] >= n_discovered else features
        free = clusterer.prototypes(pool, n_discovered, seed)
    return refine(features, np.vstack([anchors, free]), anchors)


def refine(features: sparse.csr_matrix, prototypes: np.ndarray, anchors: np.ndarray) -> np.ndarray:
    n, k = features.shape[0], len(prototypes)
    current = prototypes.copy()
    labels = None
    for _ in range(MAX_REFINE_ITERATIONS):
        new_labels = np.asarray(features @ current.T).argmax(axis=1)
        if labels is not None and np.array_equal(new_labels, labels):
            break
        labels = new_labels
        membership = sparse.csr_matrix((np.ones(n), (labels, np.arange(n))), shape=(k, n))
        means = normalize((membership @ features).toarray())
        counts = np.bincount(labels, minlength=k)
        for j in np.flatnonzero(counts):
            if j < len(anchors):
                current[j] = ANCHOR_WEIGHT * anchors[j] + (1 - ANCHOR_WEIGHT) * means[j]
            else:
                current[j] = means[j]
        current = normalize(current)
    return current
