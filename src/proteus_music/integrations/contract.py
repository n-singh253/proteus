"""Shared contract for source and destination adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from proteus_music.models.track import TrackRecord


class Capability(StrEnum):
    IMPORT = "import"
    MATCH = "match"  # Resolve canonical tracks to destination references for a preview.
    EXPORT = "export"


@dataclass(frozen=True)
class AdapterInfo:
    name: str
    capabilities: frozenset[Capability]
    description: str


@dataclass(frozen=True)
class ImportIssue:
    location: str
    reason: str


@dataclass(frozen=True)
class ImportBatch:
    """Records read from one source.

    ``failures`` were not imported; ``warnings`` were imported with caveats.
    """

    source: str
    records: tuple[TrackRecord, ...]
    failures: tuple[ImportIssue, ...] = ()
    warnings: tuple[ImportIssue, ...] = ()


@dataclass(frozen=True)
class ExportOptions:
    relative_paths: bool = False
    overwrite: bool = False


@dataclass(frozen=True)
class ExportReport:
    target: str
    files: tuple[str, ...]
    exported_tracks: int
    unexportable: tuple[ImportIssue, ...] = field(default_factory=tuple)

    @property
    def complete(self) -> bool:
        return not self.unexportable


class MatchStatus(StrEnum):
    MATCHED = "matched"
    AMBIGUOUS = "ambiguous"  # Candidates exist but none is safely the same recording.
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class TrackMatch:
    track_id: str
    status: MatchStatus
    reference: str | None  # Destination reference (e.g. a URI) when matched.
    reason: str


@dataclass(frozen=True)
class PlaylistPreview:
    index: int
    name: str
    description: str
    matches: tuple[TrackMatch, ...]

    @property
    def references(self) -> tuple[str, ...]:
        return tuple(m.reference for m in self.matches if m.reference)


@dataclass(frozen=True)
class ExportPreview:
    """What a destination export would contain; nothing has been written."""

    destination: str
    result_id: str
    playlists: tuple[PlaylistPreview, ...]

    def count(self, status: MatchStatus) -> int:
        return sum(m.status is status for p in self.playlists for m in p.matches)
