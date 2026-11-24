"""Interchangeable clustering algorithms that produce playlist prototypes."""

from __future__ import annotations

from typing import Protocol

import numpy as np
from scipy import sparse

from proteus_music.engine.clustering.kmeans import CosineKMeans


class Clusterer(Protocol):
    def prototypes(self, features: sparse.csr_matrix, k: int, seed: int) -> np.ndarray:
        """Return up to ``k`` unit-norm prototype rows for unit-norm feature rows."""
        ...


CLUSTERERS: dict[str, Clusterer] = {"spherical-kmeans": CosineKMeans()}


def get_clusterer(name: str) -> Clusterer:
    try:
        return CLUSTERERS[name]
    except KeyError:
        available = ", ".join(sorted(CLUSTERERS))
        raise ValueError(f"unknown algorithm {name!r}; available: {available}") from None
