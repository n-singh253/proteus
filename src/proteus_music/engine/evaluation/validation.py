"""Independent validation of organization results before they are saved or exported.

Every mode (exact count, range, categories, autonomous) is checked by this one function.
"""

from __future__ import annotations

from collections import Counter

from proteus_music.models.organization import OrganizationResult, PlaylistOrigin


class InvalidResultError(RuntimeError):
    """A result violates the exclusive-assignment invariants or its own constraints."""


def _membership_errors(result: OrganizationResult) -> list[str]:
    errors = []
    snapshot = set(result.snapshot_track_ids)
    if len(snapshot) != len(result.snapshot_track_ids):
        errors.append("snapshot contains duplicate track IDs")
    listed = Counter(t for p in result.playlists for t in p.track_ids)
    assigned = Counter(a.track_id for a in result.assignments)
    for name, counts in (("playlists", listed), ("assignments", assigned)):
        if duplicates := [t for t, c in counts.items() if c > 1]:
            errors.append(f"{len(duplicates)} track(s) appear more than once in {name}")
        if missing := snapshot - counts.keys():
            errors.append(f"{len(missing)} snapshot track(s) missing from {name}")
        if extra := counts.keys() - snapshot:
            errors.append(f"{len(extra)} unknown track(s) in {name}")
    membership = {t: p.index for p in result.playlists for t in p.track_ids}
    if any(membership.get(a.track_id) != a.playlist_index for a in result.assignments):
        errors.append("assignments disagree with playlist membership")
    return errors


def _constraint_errors(result: OrganizationResult) -> list[str]:
    config, playlists, errors = result.config, result.playlists, []
    low, high = config.count_bounds()
    count = len(playlists)
    if count < max(low, 1) or (high is not None and count > high):
        errors.append(f"{count} playlists is outside the allowed {low}..{high or 'any'}")
    if [p.index for p in playlists] != list(range(count)):
        errors.append("playlist indexes are not 0..n-1 in order")
    for p in playlists:
        size = len(p.track_ids)
        if size < config.min_size:
            errors.append(f"playlist {p.index} has {size} tracks, below minimum {config.min_size}")
        if config.max_size is not None and size > config.max_size:
            errors.append(f"playlist {p.index} has {size} tracks, above maximum {config.max_size}")

    def names(origin: PlaylistOrigin) -> list[str]:
        return [p.name for p in playlists if p.origin is origin]

    if sorted(names(PlaylistOrigin.REQUIRED_CATEGORY)) != sorted(
        c.name for c in config.required_categories
    ):
        errors.append("required categories are not each preserved as exactly one playlist")
    if not set(names(PlaylistOrigin.PREFERRED_CATEGORY)) <= {
        c.name for c in config.preferred_categories
    }:
        errors.append("a preferred-category playlist does not match a configured category")
    if len(names(PlaylistOrigin.UNCLASSIFIED)) > 1:
        errors.append("more than one unclassified playlist")
    return errors


def validate_result(result: OrganizationResult) -> None:
    """Check every invariant without trusting the code that produced the result."""
    errors = _membership_errors(result) + _constraint_errors(result)
    if errors:
        raise InvalidResultError("invalid organization result: " + "; ".join(errors))
