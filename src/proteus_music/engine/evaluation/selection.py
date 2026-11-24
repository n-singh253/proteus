"""Choosing a playlist count within an allowed range without exhaustive evaluation."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy import sparse

from proteus_music.engine.evaluation.metrics import SINGLE_PLAYLIST_SCORE
from proteus_music.models.quality import CountCandidate, CountSelection

SEARCH_BUDGET = 14
DENSE_PREFIX = 7
MAX_AUTO_PLAYLISTS = 40


def auto_upper_bound(
    features: sparse.csr_matrix, columns: tuple[tuple[str, str], ...], min_useful: int
) -> int:
    """Upper candidate count for autonomous mode, adapted to the collection.

    Bounded by how many practically sized playlists the analyzable tracks can fill, and by
    the available musical information: twice the number of genre/tag terms shared by at least
    a practical playlist's worth of tracks (terms combine, so groups can outnumber terms).
    """
    n = features.shape[0]
    if n == 0:
        return 1
    support = np.diff(features.tocsc().indptr)
    terms = sum(
        1
        for (block, token), count in zip(columns, support, strict=True)
        if block in ("genre", "tags") and not token.startswith("word:") and count >= min_useful
    )
    return max(1, min(MAX_AUTO_PLAYLISTS, n // min_useful, max(2, 2 * terms)))


def _coarse_grid(low: int, high: int, points: int) -> list[int]:
    grid = np.unique(np.round(np.geomspace(low, high, points)).astype(int))
    return sorted({low, high, *grid.tolist()})


def search(
    low: int, high: int, evaluate: Callable[[int], CountCandidate], budget: int = SEARCH_BUDGET
) -> list[CountCandidate]:
    """Evaluate every count in small ranges. Otherwise evaluate the smallest counts densely
    (where practical playlists usually lie), then a geometric grid refined around the best.

    Quality curves over k are not reliably unimodal, so this is a heuristic: it spends at most
    ``budget`` evaluations and reports exactly which counts it tried.
    """
    if high - low + 1 <= budget:
        return [evaluate(k) for k in range(low, high + 1)]
    dense = list(range(low, low + DENSE_PREFIX))
    coarse = _coarse_grid(dense[-1] + 1, high, (budget - DENSE_PREFIX) // 2)
    results = {k: evaluate(k) for k in (*dense, *coarse)}
    while len(results) < budget:
        best = max(results.values(), key=lambda c: c.score).playlists
        tried = sorted(results)
        position = tried.index(best)
        gaps = [tried[i] for i in (position - 1, position + 1) if 0 <= i < len(tried)]
        midpoints = [(best + g) // 2 for g in gaps if abs(best - g) > 1]
        midpoints = [m for m in midpoints if m not in results]
        if not midpoints:
            break
        for k in midpoints[: budget - len(results)]:
            results[k] = evaluate(k)
    return [results[k] for k in sorted(results)]


def choose(
    low: int,
    high: int,
    candidates: list[CountCandidate],
    sample_size: int,
    *,
    baseline: int | None = None,
) -> CountSelection:
    """Pick the best-scoring count; ties go to fewer playlists.

    ``baseline`` is the count meaning "a single musical playlist", scored at the
    single-playlist threshold instead of being clustered.
    """
    if baseline is not None:
        single = CountCandidate(
            playlists=baseline,
            score=SINGLE_PLAYLIST_SCORE,
            silhouette=None,
            fragmentation=0.0,
            low_confidence_rate=0.0,
        )
        candidates = [single, *candidates]
    best = max(candidates, key=lambda c: (round(c.score, 6), -c.playlists))
    if best.playlists == baseline:
        reason = (
            f"no split scored above the single-playlist baseline ({SINGLE_PLAYLIST_SCORE:.2f}); "
            "the analyzable metadata shows no clearly separable groups"
        )
    else:
        silhouette = "n/a" if best.silhouette is None else f"{best.silhouette:.3f}"
        reason = (
            f"{best.playlists} playlists scored highest ({best.score:.3f}: silhouette "
            f"{silhouette}, {best.fragmentation:.0%} of tracks in undersized playlists, "
            f"{best.low_confidence_rate:.0%} low-confidence) of {len(candidates)} evaluated "
            f"count(s) in {low}..{high}"
        )
    return CountSelection(
        low=low,
        high=high,
        chosen=best.playlists,
        evaluated=tuple(candidates),
        sample_size=sample_size,
        reason=reason,
    )
