"""Organization engine: descriptors and constraints in, exclusive labeled playlists out."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from proteus_music.engine.clustering.registry import get_clusterer
from proteus_music.engine.evaluation.quality import assess, cohesion
from proteus_music.engine.features.metadata import FeatureWeights, TrackDescriptor, build_features
from proteus_music.engine.naming.names import label_playlists
from proteus_music.engine.optimization.assignment import assign, check_feasibility
from proteus_music.models.organization import Assignment, OrganizationConfig, Playlist

# Tie-break weight that steers featureless tracks to the fallback playlist without
# outweighing any real similarity or hard constraint.
_FALLBACK_SCORE = 1e-6


@dataclass(frozen=True)
class PlaylistPlan:
    playlists: tuple[Playlist, ...]
    assignments: tuple[Assignment, ...]
    warnings: tuple[str, ...]
    feature_version: str


def _scores(
    descriptors: Sequence[TrackDescriptor], config: OrganizationConfig, weights: FeatureWeights
):
    features = build_features(descriptors, weights)
    k = config.playlists
    scores = np.zeros((len(descriptors), k))
    represented = features.matrix[features.available]
    if represented.shape[0] and represented.shape[1]:
        prototypes = get_clusterer(config.algorithm).prototypes(represented, k, config.seed)
        scores[:, : len(prototypes)] = features.matrix @ prototypes.T
    else:
        get_clusterer(config.algorithm)  # Still reject unknown algorithms.
        prototypes = np.zeros((0, 0))

    warnings = []
    if len(prototypes) < k:
        warnings.append(
            f"only {len(prototypes)} distinct musical profile(s) found for {k} playlists; "
            f"{k - len(prototypes)} playlist(s) are filled only to meet the requested count"
        )
    missing = ~features.available
    if missing.any():
        fallback = int(np.bincount(scores[features.available].argmax(axis=1), minlength=k).argmax())
        scores[missing, fallback] = _FALLBACK_SCORE
        warnings.append(
            f"{int(missing.sum())} track(s) have no analyzable metadata; they are placed in the "
            "largest playlist where constraints allow and flagged missing-features"
        )
    return scores, features, warnings


def plan_playlists(
    descriptors: Sequence[TrackDescriptor],
    config: OrganizationConfig,
    weights: FeatureWeights = FeatureWeights(),
) -> PlaylistPlan:
    check_feasibility(len(descriptors), config)
    scores, features, warnings = _scores(descriptors, config, weights)
    labels = assign(scores, config.min_size, config.max_size)
    qualities = assess(scores, labels, features.available)
    cohesions = cohesion(scores, labels, features.available)

    groups = []
    for j in range(config.playlists):
        members = np.flatnonzero(labels == j)
        groups.append(members[np.argsort(-scores[members, j], kind="stable")])
    names = label_playlists([[descriptors[i] for i in g] for g in groups])

    playlists = tuple(
        Playlist(
            index=j,
            name=names[j].name,
            description=names[j].description,
            track_ids=tuple(descriptors[i].track_id for i in group),
            cohesion=cohesions[j],
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
    return PlaylistPlan(playlists, assignments, tuple(warnings), features.version)
