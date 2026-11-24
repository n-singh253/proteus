"""Assignment confidence and playlist cohesion from song-to-playlist similarities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from proteus_music.models.organization import AssignmentFlag

LOW_SIMILARITY = 0.15
AMBIGUOUS_MARGIN = 0.02


@dataclass(frozen=True)
class AssignmentQuality:
    similarity: float
    margin: float
    flags: tuple[AssignmentFlag, ...]


def assess(
    scores: np.ndarray, labels: np.ndarray, available: np.ndarray
) -> list[AssignmentQuality]:
    """Similarity to the assigned playlist, margin over the best alternative, and flags."""
    n, k = scores.shape
    rows = np.arange(n)
    chosen = scores[rows, labels]
    others = scores.copy()
    others[rows, labels] = -np.inf
    best_other = others.max(axis=1) if k > 1 else np.zeros(n)
    margins = chosen - best_other
    qualities = []
    for similarity, margin, has_features in zip(chosen, margins, available, strict=True):
        flags: list[AssignmentFlag] = []
        if not has_features:
            flags.append(AssignmentFlag.MISSING_FEATURES)
        else:
            if similarity < LOW_SIMILARITY:
                flags.append(AssignmentFlag.LOW_SIMILARITY)
            if margin < 0:
                flags.append(AssignmentFlag.CONSTRAINT_DISPLACED)
            elif margin < AMBIGUOUS_MARGIN:
                flags.append(AssignmentFlag.AMBIGUOUS)
        qualities.append(AssignmentQuality(float(similarity), float(margin), tuple(flags)))
    return qualities


def cohesion(scores: np.ndarray, labels: np.ndarray, available: np.ndarray) -> list[float]:
    """Mean similarity of each playlist's analyzable members to its prototype (0 if none)."""
    k = scores.shape[1]
    values = []
    for j in range(k):
        members = (labels == j) & available
        values.append(float(scores[members, j].mean()) if members.any() else 0.0)
    return values
