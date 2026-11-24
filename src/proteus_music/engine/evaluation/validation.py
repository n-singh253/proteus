"""Independent validation of organization results before they are saved or exported."""

from __future__ import annotations

from collections import Counter

from proteus_music.models.organization import OrganizationResult


class InvalidResultError(RuntimeError):
    """A result violates the exclusive-assignment invariants or its own constraints."""


def validate_result(result: OrganizationResult) -> None:
    """Check every invariant without trusting the code that produced the result."""
    config, errors = result.config, []
    snapshot = set(result.snapshot_track_ids)
    if len(snapshot) != len(result.snapshot_track_ids):
        errors.append("snapshot contains duplicate track IDs")
    if len(result.playlists) != config.playlists:
        errors.append(f"expected {config.playlists} playlists, found {len(result.playlists)}")
    if [p.index for p in result.playlists] != list(range(len(result.playlists))):
        errors.append("playlist indexes are not 0..n-1 in order")

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
    for p in result.playlists:
        size = len(p.track_ids)
        if size < config.min_size:
            errors.append(f"playlist {p.index} has {size} tracks, below minimum {config.min_size}")
        if config.max_size is not None and size > config.max_size:
            errors.append(f"playlist {p.index} has {size} tracks, above maximum {config.max_size}")
    if errors:
        raise InvalidResultError("invalid organization result: " + "; ".join(errors))
