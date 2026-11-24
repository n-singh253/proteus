"""Exclusive song-to-playlist assignment under playlist count and size constraints."""

from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.optimize import Bounds, LinearConstraint, linprog, milp

from proteus_music.models.organization import OrganizationConfig

_INTEGRAL_TOLERANCE = 1e-6


class InfeasibleConstraintsError(ValueError):
    """The requested hard constraints cannot all be satisfied."""

    def __init__(self, reasons: list[str]) -> None:
        super().__init__("; ".join(reasons))
        self.reasons = reasons


def check_feasibility(n_tracks: int, config: OrganizationConfig) -> None:
    k, low, high = config.playlists, config.min_size, config.max_size
    reasons = []
    if n_tracks == 0:
        reasons.append("the library has no tracks; import a collection first")
    elif n_tracks < k * low:
        reasons.append(
            f"{k} playlists with at least {low} track(s) each need {k * low} tracks, "
            f"but the library has {n_tracks}"
        )
    if high is not None and n_tracks > k * high:
        reasons.append(
            f"{k} playlists with at most {high} tracks each hold {k * high} tracks, "
            f"but the library has {n_tracks}"
        )
    if reasons:
        raise InfeasibleConstraintsError(reasons)


def assign(scores: np.ndarray, min_size: int, max_size: int | None) -> np.ndarray:
    """Maximize total score with each row in exactly one column and column sizes in bounds.

    Returns the column index per row. The independent best-score assignment is used when it
    already satisfies the bounds. Otherwise this is a transportation problem whose constraint
    matrix is totally unimodular, so an LP vertex solution is integral; HiGHS interior point
    with crossover returns such a vertex far faster than branch-and-bound. Exact MILP remains
    as a fallback should the LP ever return a fractional point.
    """
    n, k = scores.shape
    high = n if max_size is None else max_size
    labels = np.argmax(scores, axis=1)
    sizes = np.bincount(labels, minlength=k)
    if sizes.min() >= min_size and sizes.max() <= high:
        return labels

    # Variable x[i, j] lives at column i * k + j.
    rows = sparse.kron(sparse.eye(n, format="csr"), np.ones((1, k)), format="csr")
    cols = sparse.kron(np.ones((1, n)), sparse.eye(k, format="csr"), format="csr")
    lp = linprog(
        -scores.ravel(),
        A_ub=sparse.vstack([cols, -cols], format="csr"),
        b_ub=np.r_[np.full(k, high), np.full(k, -min_size)],
        A_eq=rows,
        b_eq=np.ones(n),
        bounds=(0, 1),
        method="highs-ipm",
    )
    if lp.status == 0 and np.all(np.minimum(lp.x, 1 - lp.x) < _INTEGRAL_TOLERANCE):
        return np.argmax(lp.x.reshape(n, k), axis=1)

    exact = milp(
        c=-scores.ravel(),
        constraints=[LinearConstraint(rows, 1, 1), LinearConstraint(cols, min_size, high)],
        integrality=np.ones(n * k),
        bounds=Bounds(0, 1),
    )
    if not exact.success:
        raise InfeasibleConstraintsError([f"assignment solver failed: {exact.message}"])
    return np.argmax(exact.x.reshape(n, k), axis=1)
