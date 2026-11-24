"""Library import and inspection services shared by the CLI and API."""

from __future__ import annotations

import sqlite3
import uuid
from collections import Counter
from dataclasses import dataclass

from proteus_music.integrations.contract import ImportBatch, ImportIssue
from proteus_music.library.identity import IdentityIndex
from proteus_music.storage import tracks as track_store


@dataclass(frozen=True)
class ImportSummary:
    source: str
    records_read: int
    new_tracks: int
    matched_existing: int
    refreshed_records: int
    failures: tuple[ImportIssue, ...]
    warnings: tuple[ImportIssue, ...]


@dataclass(frozen=True)
class LibraryStats:
    tracks: int
    source_records: int
    records_by_source: dict[str, int]
    tracks_with_local_file: int
    tracks_without_analysis_data: int


def import_batch(conn: sqlite3.Connection, batch: ImportBatch) -> ImportSummary:
    """Resolve records to canonical tracks and persist them in one transaction."""
    index = IdentityIndex.from_tracks(track_store.load_tracks(conn))
    new = matched = refreshed = 0
    with conn:
        for record in batch.records:
            track_id = track_store.record_track_id(conn, record.source, record.source_key)
            if track_id is not None:
                refreshed += 1
            else:
                track_id = index.resolve(record)
                if track_id is None:
                    track_id = uuid.uuid4().hex
                    track_store.insert_track(conn, track_id, record)
                    new += 1
                else:
                    track_store.fill_missing_fields(conn, track_id, record)
                    matched += 1
            track_store.save_record(conn, track_id, record)
            index.add(track_id, record)
    return ImportSummary(
        source=batch.source,
        records_read=len(batch.records),
        new_tracks=new,
        matched_existing=matched,
        refreshed_records=refreshed,
        failures=batch.failures,
        warnings=batch.warnings,
    )


def library_stats(conn: sqlite3.Connection) -> LibraryStats:
    tracks = track_store.load_tracks(conn)
    by_source = Counter(r.source for t in tracks for r in t.records)
    return LibraryStats(
        tracks=len(tracks),
        source_records=sum(by_source.values()),
        records_by_source=dict(sorted(by_source.items())),
        tracks_with_local_file=sum(1 for t in tracks if t.local_path),
        tracks_without_analysis_data=sum(1 for t in tracks if not t.analysis_records),
    )
