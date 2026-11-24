"""Reusable quality evaluation of an organization, independent of how it was produced.

Inputs are the feature rows, labels and assignment similarities, so any future clustering
algorithm or representation can be scored the same way. No pairwise matrix is materialized
beyond a bounded silhouette sample.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize

from proteus_music.models.quality import QualityReport

SILHOUETTE_SAMPLE = 2000
# Selection objective: silhouette minus penalties for impractical results. A single playlist
# scores SINGLE_PLAYLIST_SCORE, so a split must show at least that much structure to win.
FRAGMENTATION_PENALTY = 0.5
LOW_CONFIDENCE_PENALTY = 0.25
SINGLE_PLAYLIST_SCORE = 0.12


def practical_min_size(n_tracks: int) -> int:
    """Smallest playlist still worth listening to on its own: about an hour (15 tracks),
    growing to 25 for large libraries where finer splits add little."""
    return max(15, min(25, n_tracks // 40))


def selection_score(silhouette: float | None, fragmentation: float, low_confidence: float) -> float:
    base = SINGLE_PLAYLIST_SCORE if silhouette is None else silhouette
    return base - FRAGMENTATION_PENALTY * fragmentation - LOW_CONFIDENCE_PENALTY * low_confidence


def _silhouette(features: sparse.csr_matrix, labels: np.ndarray, seed: int) -> float | None:
    n = features.shape[0]
    if len(np.unique(labels)) < 2 or n < 3:
        return None
    sample = min(n, SILHOUETTE_SAMPLE)
    try:
        value = silhouette_score(
            features, labels, metric="cosine", sample_size=sample, random_state=seed
        )
    except ValueError:  # The sample happened to contain a single label.
        return None
    return float(value)


def _semantic_coherence(membership: sparse.csr_matrix, terms: sparse.csr_matrix) -> float:
    """Size-weighted share of members carrying their playlist's most common genre or tag."""
    total = membership.shape[1]
    if not total or not terms.shape[1]:
        return 0.0
    counts = (membership @ terms).toarray()
    return float(counts.max(axis=1).sum() / total)


def evaluate(
    features: sparse.csr_matrix,
    available: np.ndarray,
    labels: np.ndarray,
    similarity: np.ndarray,
    flagged: np.ndarray,
    terms: sparse.csr_matrix,
    n_playlists: int,
    *,
    unclassified: int | None,
    min_useful_share: float,
    seed: int,
) -> QualityReport:
    """Score an organization. ``terms`` is a binary track x descriptive-term matrix (see
    ``term_presence``); ``min_useful_share`` is the playlist share below which a playlist
    counts as fragmentary; the unclassified playlist is excluded from clustering metrics."""
    n = len(labels)
    sizes = np.bincount(labels, minlength=n_playlists)
    real = [j for j in range(n_playlists) if j != unclassified]
    analyzed = available & (labels != (-1 if unclassified is None else unclassified))
    x, y = features[analyzed], labels[analyzed]

    membership = sparse.csr_matrix(
        (np.ones(len(y)), (y, np.arange(len(y)))), shape=(n_playlists, len(y))
    )
    sums = (membership @ x).toarray() if len(y) else np.zeros((n_playlists, features.shape[1]))
    present = [j for j in real if np.any(y == j)]
    cohesion = float(np.linalg.norm(sums, axis=1).sum() / len(y)) if len(y) else 0.0
    separation = None
    if len(present) > 1:
        centroids = normalize(sums[present])
        pairwise = centroids @ centroids.T
        off_diagonal = pairwise[~np.eye(len(present), dtype=bool)]
        separation = float(1 - off_diagonal.mean())

    undersized = [j for j in real if sizes[j] < min_useful_share * n]
    fragmentation = float(sizes[undersized].sum() / n) if n else 0.0
    low_confidence = float(flagged.mean()) if n else 0.0
    silhouette = _silhouette(x, y, seed)
    return QualityReport(
        silhouette=silhouette,
        cohesion=cohesion,
        separation=separation,
        compatibility=float(similarity[analyzed].mean()) if len(y) else 0.0,
        smallest_playlist=int(sizes[real].min()) if real else 0,
        largest_playlist=int(sizes[real].max()) if real else 0,
        fragmentation=fragmentation,
        low_confidence_rate=low_confidence,
        semantic_coherence=_semantic_coherence(membership, terms[analyzed]),
        analyzable_share=float(available.mean()) if n else 0.0,
        score=selection_score(silhouette, fragmentation, low_confidence),
    )
