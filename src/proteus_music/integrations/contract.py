"""Shared contract for source and destination adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from proteus_music.models.track import TrackRecord


class Capability(StrEnum):
    IMPORT = "import"
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
