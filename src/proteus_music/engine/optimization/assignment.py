"""Exclusive song-to-playlist assignment under playlist count and size constraints."""

from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.optimize import Bounds, LinearConstraint, linprog, milp

from proteus_music.models.organization import OrganizationConfig

_INTEGRAL_TOLERANCE = 1e-6
_REDUCED_COST_TOLERANCE = 1e-9
_MAX_ROUNDS = 30
CANDIDATES_PER_TRACK = 4


class InfeasibleConstraintsError(ValueError):
    """The requested hard constraints cannot all be satisfied."""

    def __init__(self, reasons: list[str]) -> None:
        super().__init__("; ".join(reasons))
        self.reasons = reasons


def check_feasibility(n_tracks: int, config: OrganizationConfig) -> tuple[int, int]:
    """Feasible total playlist counts (inclusive) for the configuration's hard constraints.

    Runs before any feature or clustering work so impossible requests fail immediately.
    """
    low, high = config.min_size, config.max_size
    if n_tracks == 0:
        raise InfeasibleConstraintsError(["the library has no tracks; import a collection first"])
    if config.playlists is not None:
        k, reasons = config.playlists, []
        if n_tracks < k * low:
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
        return k, k

    bound_low, bound_high = config.count_bounds()
    size_low = -(-n_tracks // high) if high is not None else 1
    size_high = n_tracks // low
    feasible_low = max(bound_low, size_low, 1)
    feasible_high = size_high if bound_high is None else min(bound_high, size_high)
    if feasible_low > feasible_high:
        sizes = f"{low}..{high if high is not None else 'unbounded'}"
        wanted = f"{bound_low}..{bound_high if bound_high is not None else 'any'}"
        raise InfeasibleConstraintsError(
            [
                f"no playlist count in {wanted} works for {n_tracks} tracks with playlist "
                f"sizes {sizes}: those sizes allow {size_low}..{size_high} playlists"
            ]
        )
    return feasible_low, feasible_high


def _initial_columns(scores: np.ndarray, min_size: int, per_track: int) -> np.ndarray:
    """Each track's best ``per_track`` playlists, plus each playlist's best tracks so that
    minimum sizes are reachable in the restricted problem."""
    n, k = scores.shape
    active = np.zeros((n, k), dtype=bool)
    if per_track >= k:
        active[:] = True
        return active
    top = np.argpartition(-scores, per_track - 1, axis=1)[:, :per_track]
    active[np.arange(n)[:, None], top] = True
    need = min(n, 2 * min_size)
    best = np.argpartition(-scores, need - 1, axis=0)[:need]
    active[best, np.arange(k)] = True
    return active


def _restricted_lp(scores: np.ndarray, active: np.ndarray, min_size: int, high: int):
    n, k = scores.shape
    rows, cols = np.nonzero(active)
    variables = np.arange(len(rows))
    ones = np.ones(len(rows))
    per_track = sparse.csr_matrix((ones, (rows, variables)), shape=(n, len(rows)))
    per_playlist = sparse.csr_matrix((ones, (cols, variables)), shape=(k, len(rows)))
    result = linprog(
        -scores[rows, cols],
        A_ub=sparse.vstack([per_playlist, -per_playlist], format="csr"),
        b_ub=np.r_[np.full(k, high), np.full(k, -min_size)],
        A_eq=per_track,
        b_eq=np.ones(n),
        bounds=(0, 1),
        method="highs-ipm",
    )
    return result, rows, cols


def _column_generation(scores: np.ndarray, min_size: int, high: int) -> np.ndarray | None:
    """Solve the transportation LP over a growing subset of (track, playlist) variables.

    The restricted optimum is globally optimal once no excluded variable has a negative
    reduced cost under the LP duals; violating variables are added and the LP re-solved.
    Returns None if this does not finish with an integral solution.
    """
    k = scores.shape[1]
    per_track = min(k, CANDIDATES_PER_TRACK)
    active = _initial_columns(scores, min_size, per_track)
    for _ in range(_MAX_ROUNDS):
        result, rows, cols = _restricted_lp(scores, active, min_size, high)
        if result.status == 2 and not active.all():  # Restricted problem infeasible: widen.
            per_track = min(k, 2 * per_track)
            active |= _initial_columns(scores, min_size, per_track)
            continue
        if result.status != 0:
            return None
        duals = result.ineqlin.marginals
        reduced = -scores - result.eqlin.marginals[:, None] - (duals[:k] - duals[k:])[None, :]
        violating = ~active & (reduced < -_REDUCED_COST_TOLERANCE)
        if violating.any():
            active |= violating
            continue
        if np.any(np.minimum(result.x, 1 - result.x) >= _INTEGRAL_TOLERANCE):
            return None
        labels = np.empty(scores.shape[0], dtype=np.int64)
        chosen = result.x > 0.5
        labels[rows[chosen]] = cols[chosen]
        return labels
    return None


def assign(scores: np.ndarray, min_size: int, max_size: int | None) -> np.ndarray:
    """Maximize total score with each row in exactly one column and column sizes in bounds.

    Returns the column index per row. The independent best-score assignment is used when it
    already satisfies the bounds. Otherwise this is a transportation problem whose constraint
    matrix is totally unimodular, so an LP vertex solution is integral; HiGHS interior point
    with crossover returns such a vertex, and column generation keeps the LP small for large
    libraries without giving up exactness. Exact MILP remains as a fallback.
    """
    n, k = scores.shape
    high = n if max_size is None else max_size
    labels = np.argmax(scores, axis=1)
    sizes = np.bincount(labels, minlength=k)
    if sizes.min() >= min_size and sizes.max() <= high:
        return labels
    solved = _column_generation(scores, min_size, high)
    if solved is not None:
        return solved

    # Variable x[i, j] lives at column i * k + j.
    rows = sparse.kron(sparse.eye(n, format="csr"), np.ones((1, k)), format="csr")
    cols = sparse.kron(np.ones((1, n)), sparse.eye(k, format="csr"), format="csr")
    exact = milp(
        c=-scores.ravel(),
        constraints=[LinearConstraint(rows, 1, 1), LinearConstraint(cols, min_size, high)],
        integrality=np.ones(n * k),
        bounds=Bounds(0, 1),
    )
    if not exact.success:
        raise InfeasibleConstraintsError([f"assignment solver failed: {exact.message}"])
    return np.argmax(exact.x.reshape(n, k), axis=1)
