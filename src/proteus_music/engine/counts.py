"""Playlist-count planning: the feasible count range, the Unclassified decision, and
sample-based selection of a count within a range."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy import sparse

from proteus_music.engine.classification.categories import ResolvedCategories
from proteus_music.engine.clustering.registry import Clusterer
from proteus_music.engine.clustering.seeded import build_prototypes
from proteus_music.engine.evaluation import metrics, selection
from proteus_music.engine.evaluation.quality import assess
from proteus_music.engine.features.metadata import FeatureMatrix, TrackDescriptor, term_presence
from proteus_music.engine.optimization.assignment import assign
from proteus_music.models.organization import OrganizationConfig, OrganizationMode
from proteus_music.models.quality import CountCandidate, CountSelection

EVALUATION_SAMPLE = 6000


@dataclass(frozen=True)
class Run:
    config: OrganizationConfig
    descriptors: Sequence[TrackDescriptor]
    features: FeatureMatrix
    categories: ResolvedCategories
    clusterer: Clusterer

    @property
    def n_fixed(self) -> int:
        return len(self.categories.categories)


@dataclass(frozen=True)
class CountRange:
    """Bounds on feature-based playlists (categories + discovered), plus Unclassified."""

    low: int
    high: int
    unclassified: bool


def similarity_scores(x: sparse.csr_matrix, prototypes: np.ndarray, k: int) -> np.ndarray:
    scores = np.zeros((x.shape[0], k))
    if len(prototypes):
        scores[:, : len(prototypes)] = x @ prototypes.T
    return scores


def count_range(run: Run, low: int, high: int) -> CountRange:
    """Narrow the feasible total range to what this mode and collection support.

    An Unclassified playlist is used only when the count is flexible (auto or range), it
    fits the size constraints, and the remaining analyzable tracks still fit.
    """
    config, available = run.config, run.features.available
    n, n_analyzable = len(available), int(available.sum())
    if config.playlists is not None:
        return CountRange(low, high, False)
    if config.mode is not OrganizationMode.AUTO and not config.has_range:
        return CountRange(run.n_fixed, run.n_fixed, False)  # One playlist per category.
    if config.mode is OrganizationMode.AUTO or config.max_playlists is None:
        useful = metrics.practical_min_size(n)
        cap = selection.auto_upper_bound(
            run.features.matrix[run.features.core], run.features.columns, useful
        )
        high = max(low, min(high, max(cap, run.n_fixed)))
    low = max(low, run.n_fixed)
    missing = n - n_analyzable
    fits = missing >= config.min_size and (config.max_size is None or missing <= config.max_size)
    if missing and fits:
        if n_analyzable == 0 and low <= 1:
            return CountRange(0, 0, True)
        by_size = -(-n_analyzable // config.max_size) if config.max_size else 1
        k_low = max(low - 1, run.n_fixed, 1, by_size)
        k_high = min(high - 1, n_analyzable // config.min_size)
        if n_analyzable and k_low <= k_high:
            return CountRange(k_low, k_high, True)
    return CountRange(low, high, False)


def select_count(run: Run, counts: CountRange) -> CountSelection:
    """Evaluate candidate counts on a seeded sample of analyzable tracks.

    Size constraints are scaled to the sample, so candidates are judged under the same kind
    of constrained assignment the final run uses; the final run then uses every track.
    """
    config, features = run.config, run.features
    rows = np.flatnonzero(features.core)
    n_constrained = (
        int(features.available.sum()) if counts.unclassified else len(features.available)
    )
    if len(rows) > EVALUATION_SAMPLE:
        rng = np.random.default_rng(config.seed)
        rows = np.sort(rng.choice(rows, EVALUATION_SAMPLE, replace=False))
    x, s = features.matrix[rows], len(rows)
    min_s = max(1, config.min_size * s // n_constrained)
    max_s = None if config.max_size is None else -(-config.max_size * s // n_constrained)
    terms = term_presence(features)[rows]
    every = np.ones(s, dtype=bool)
    n_total = len(features.available)
    useful_share = metrics.practical_min_size(n_total) / n_total
    extra = int(counts.unclassified)

    def evaluate_count(k: int) -> CountCandidate:
        prototypes = build_prototypes(
            x, run.categories.prototypes, k - run.n_fixed, run.clusterer, config.seed
        )
        scores = similarity_scores(x, prototypes, k)
        labels = assign(scores, min(min_s, s // k), max_s)
        qualities = assess(scores, labels, every)
        report = metrics.evaluate(
            x,
            every,
            labels,
            np.array([q.similarity for q in qualities]),
            np.array([bool(q.flags) for q in qualities]),
            terms,
            k,
            unclassified=None,
            min_useful_share=useful_share,
            seed=config.seed,
        )
        return CountCandidate(
            playlists=k + extra,
            score=report.score,
            silhouette=report.silhouette,
            fragmentation=report.fragmentation,
            low_confidence_rate=report.low_confidence_rate,
        )

    single = counts.low == 1 and run.n_fixed == 0
    start = max(counts.low, 2) if single else counts.low
    evaluated = []
    if start <= counts.high:
        evaluated = selection.search(
            start + extra, counts.high + extra, lambda t: evaluate_count(t - extra)
        )
    return selection.choose(
        counts.low + extra,
        counts.high + extra,
        evaluated,
        s,
        baseline=1 + extra if single else None,
    )
