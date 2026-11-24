"""Organization engine: descriptors and constraints in, exclusive labeled playlists out.

Every mode follows one path: feasibility -> features -> categories -> count range ->
(count selection) -> prototypes -> constrained assignment -> evaluation -> naming.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, replace

import numpy as np

from proteus_music.engine.classification.categories import resolve_categories
from proteus_music.engine.clustering.registry import get_clusterer
from proteus_music.engine.clustering.seeded import build_prototypes
from proteus_music.engine.counts import Run, count_range, select_count, similarity_scores
from proteus_music.engine.evaluation import metrics
from proteus_music.engine.evaluation.quality import AssignmentQuality, assess, cohesion
from proteus_music.engine.features.metadata import (
    FeatureWeights,
    TrackDescriptor,
    build_features,
    term_presence,
)
from proteus_music.engine.naming.names import label_playlists
from proteus_music.engine.optimization.assignment import assign, check_feasibility
from proteus_music.models.organization import (
    Assignment,
    AssignmentFlag,
    OrganizationConfig,
    Playlist,
    PlaylistOrigin,
)
from proteus_music.models.quality import CategoryReport, CountSelection, Evidence, QualityReport

UNCLASSIFIED_NAME = "Unclassified"
# Tie-break weight that steers featureless tracks to the fallback playlist without
# outweighing any real similarity or hard constraint.
_FALLBACK_SCORE = 1e-6
_MISSING = AssignmentQuality(0.0, 0.0, (AssignmentFlag.MISSING_FEATURES,))


@dataclass(frozen=True)
class PlaylistPlan:
    playlists: tuple[Playlist, ...]
    assignments: tuple[Assignment, ...]
    warnings: tuple[str, ...]
    feature_version: str
    quality: QualityReport
    selection: CountSelection | None
    categories: tuple[CategoryReport, ...]
    timings: dict[str, float]


def _assign(run: Run, prototypes: np.ndarray, k: int, unclassified: bool, warnings: list[str]):
    """Labels (Unclassified = index k), similarity scores, and per-track quality."""
    config, available, core = run.config, run.features.available, run.features.core
    n = len(available)
    scores = similarity_scores(run.features.matrix, prototypes, k)
    if unclassified:
        labels = np.full(n, k)
        rows = np.flatnonzero(available)
        qualities: list[AssignmentQuality] = []
        if k:
            labels[rows] = assign(scores[rows], config.min_size, config.max_size)
            qualities = assess(scores[rows], labels[rows], available[rows], core[rows])
        by_row = dict(zip(rows.tolist(), qualities, strict=True))
        warnings.append(
            f"{n - len(rows)} track(s) without analyzable metadata were placed in "
            f"'{UNCLASSIFIED_NAME}' rather than guessed into a musical group"
        )
        return labels, scores, [by_row.get(i, _MISSING) for i in range(n)]
    missing = ~available
    if missing.any():
        fallback = int(np.bincount(scores[available].argmax(axis=1), minlength=k).argmax())
        scores[missing, fallback] = _FALLBACK_SCORE
        warnings.append(
            f"{int(missing.sum())} track(s) have no analyzable metadata; they are placed in the "
            "largest playlist where constraints allow and flagged missing-features"
        )
    labels = assign(scores, config.min_size, config.max_size)
    return labels, scores, assess(scores, labels, available, core)


def _information_warnings(run: Run, prototypes: np.ndarray, k: int) -> list[str]:
    warnings = []
    discovered, requested = len(prototypes) - run.n_fixed, k - run.n_fixed
    if k and discovered < requested:
        warnings.append(
            f"only {max(discovered, 0)} distinct musical profile(s) found for {requested} "
            f"discovered playlist(s); the rest are filled only to meet the count"
        )
    partial = int((run.features.available & ~run.features.core).sum())
    if partial:
        warnings.append(
            f"{partial} track(s) have only artist or release-year metadata; they were matched "
            "to playlists shaped by fuller metadata and flagged partial-features"
        )
    share = float(run.features.available.mean())
    if share == 0:
        warnings.append("no track has permitted analyzable metadata; playlists carry no meaning")
    elif share < 0.5:
        warnings.append(
            f"only {share:.0%} of tracks have permitted analyzable metadata; the organization "
            "reflects that subset"
        )
    for report in run.categories.reports:
        if report.kept and report.evidence is Evidence.WEAK:
            warnings.append(f"category '{report.name}' has weak evidence: {report.note}")
        elif not report.kept:
            warnings.append(f"preferred category '{report.name}' not created: {report.note}")
    return warnings


def _build(run: Run, labels, scores, qualities, k: int, unclassified: bool):
    features, descriptors = run.features, run.descriptors
    n, total = len(labels), k + int(unclassified)
    groups = []
    for j in range(total):
        members = np.flatnonzero(labels == j)
        if j < k:
            members = members[np.argsort(-scores[members, j], kind="stable")]
        groups.append(members)
    fixed = {j: c.name for j, c in enumerate(run.categories.categories)}
    origins = [*run.categories.origins, *[PlaylistOrigin.DISCOVERED] * (k - run.n_fixed)]
    if unclassified:
        fixed[k] = UNCLASSIFIED_NAME
        origins.append(PlaylistOrigin.UNCLASSIFIED)
    names = label_playlists([[descriptors[i] for i in g] for g in groups], fixed)
    cohesions = [*cohesion(scores, labels, features.core), *[0.0] * int(unclassified)]
    playlists = tuple(
        Playlist(
            index=j,
            name=names[j].name,
            description=names[j].description,
            track_ids=tuple(descriptors[i].track_id for i in group),
            cohesion=cohesions[j],
            origin=origins[j],
        )
        for j, group in enumerate(groups)
    )
    assignments = tuple(
        Assignment(
            track_id=d.track_id,
            playlist_index=int(labels[i]),
            similarity=qualities[i].similarity,
            margin=qualities[i].margin,
            flags=qualities[i].flags,
        )
        for i, d in enumerate(descriptors)
    )
    quality = metrics.evaluate(
        features.matrix,
        features.core,
        labels,
        np.array([q.similarity for q in qualities]),
        np.array([bool(q.flags) for q in qualities]),
        term_presence(features),
        total,
        unclassified=k if unclassified else None,
        min_useful_share=metrics.practical_min_size(n) / n,
        seed=run.config.seed,
    )
    return playlists, assignments, quality


def plan_playlists(
    descriptors: Sequence[TrackDescriptor],
    config: OrganizationConfig,
    weights: FeatureWeights = FeatureWeights(),
) -> PlaylistPlan:
    clock, timings = time.perf_counter, {}
    low, high = check_feasibility(len(descriptors), config)
    clusterer = get_clusterer(config.algorithm)
    start = clock()
    features = build_features(descriptors, weights)
    resolved = resolve_categories(
        config.categories,
        features,
        weights,
        min_size=config.min_size,
        preferred_slots=high - len(config.required_categories),
    )
    run = Run(config, descriptors, features, resolved, clusterer)
    counts = count_range(run, low, high)
    timings["features"] = clock() - start

    chosen, k = None, counts.low
    if counts.high > counts.low and features.available.any():
        start = clock()
        chosen = select_count(run, counts)
        k = chosen.chosen - int(counts.unclassified)
        timings["selection"] = clock() - start

    run = replace(run, categories=resolved.for_count(k))
    start = clock()
    analyzable = features.matrix[features.core]
    prototypes = np.zeros((0, features.matrix.shape[1]))
    if k and analyzable.shape[0] and analyzable.shape[1]:
        prototypes = build_prototypes(
            analyzable, run.categories.prototypes, k - run.n_fixed, clusterer, config.seed
        )
    timings["clustering"] = clock() - start

    start = clock()
    warnings = _information_warnings(run, prototypes, k)
    labels, scores, qualities = _assign(run, prototypes, k, counts.unclassified, warnings)
    weak = sum(AssignmentFlag.LOW_SIMILARITY in q.flags for q in qualities)
    if weak > 0.1 * len(qualities):
        warnings.append(
            f"{weak} track(s) fit no playlist well (flagged low-similarity); more playlists "
            "or categories may describe them better"
        )
    timings["assignment"] = clock() - start

    start = clock()
    playlists, assignments, quality = _build(run, labels, scores, qualities, k, counts.unclassified)
    timings["evaluation"] = clock() - start
    return PlaylistPlan(
        playlists,
        assignments,
        tuple(warnings),
        features.version,
        quality,
        chosen,
        run.categories.reports,
        timings,
    )
