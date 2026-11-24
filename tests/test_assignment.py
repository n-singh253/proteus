import itertools

import numpy as np
import pytest

from proteus_music.engine.optimization.assignment import (
    InfeasibleConstraintsError,
    assign,
    check_feasibility,
)
from proteus_music.models.organization import OrganizationConfig


def test_constrained_assignment_is_optimal_and_respects_sizes():
    rng = np.random.default_rng(7)
    scores = rng.random((7, 3))
    scores[:, 0] += 1.0  # Unconstrained, every track would pick playlist 0.

    labels = assign(scores, min_size=2, max_size=3)

    sizes = np.bincount(labels, minlength=3)
    assert labels.shape == (7,) and sizes.min() >= 2 and sizes.max() <= 3
    best = max(
        sum(scores[i, j] for i, j in enumerate(combo))
        for combo in itertools.product(range(3), repeat=7)
        if all(2 <= combo.count(j) <= 3 for j in range(3))
    )
    assert scores[np.arange(7), labels].sum() == pytest.approx(best)


@pytest.mark.parametrize(
    ("n_tracks", "config", "message"),
    [
        (5, OrganizationConfig(playlists=10), "need 10 tracks"),
        (30, OrganizationConfig(playlists=2, max_size=10), "hold 20 tracks"),
        (0, OrganizationConfig(playlists=1), "no tracks"),
    ],
)
def test_infeasible_configurations_are_rejected(n_tracks, config, message):
    with pytest.raises(InfeasibleConstraintsError, match=message):
        check_feasibility(n_tracks, config)
