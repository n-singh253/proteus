"""Persistence for canonical tracks and their source records."""

from __future__ import annotations

import json
import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime

from proteus_music.library.identity import parse_title
from proteus_music.models.track import Track, TrackRecord


def _now() -> str:
    return datetime.now(UTC).isoformat()


def find_record(
    conn: sqlite3.Connection, source: str, source_key: str
) -> tuple[str, TrackRecord] | None:
    """The canonical track ID and stored record for a source reference, if imported before."""
    row = conn.execute(
        "SELECT track_id, record FROM source_records WHERE source = ? AND source_key = ?",
        (source, source_key),
    ).fetchone()
    return (row["track_id"], TrackRecord.model_validate_json(row["record"])) if row else None


_CANONICAL = "title, artists, album, duration_ms, year, isrc, musicbrainz_recording_id, version"


def _canonical(record: TrackRecord) -> tuple:
    parts = parse_title(record.title)
    artists = list(dict.fromkeys([*record.artists, *parts.featured]))
    return (
        record.title,
        json.dumps(artists),
        record.album,
        record.duration_ms,
        record.year,
        record.isrc,
        record.musicbrainz_recording_id,
        parts.version,
    )


def insert_track(conn: sqlite3.Connection, track_id: str, record: TrackRecord) -> None:
    conn.execute(
        f"INSERT INTO tracks (id, {_CANONICAL}, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (track_id, *_canonical(record), _now()),
    )


def fill_missing_fields(conn: sqlite3.Connection, track_id: str, record: TrackRecord) -> None:
    """Complete a canonical track's empty fields from a newly matched record; never overwrite."""
    conn.execute(
        "UPDATE tracks SET album = COALESCE(album, ?), duration_ms = COALESCE(duration_ms, ?),"
        " year = COALESCE(year, ?), isrc = COALESCE(isrc, ?),"
        " musicbrainz_recording_id = COALESCE(musicbrainz_recording_id, ?) WHERE id = ?",
        (
            record.album,
            record.duration_ms,
            record.year,
            record.isrc,
            record.musicbrainz_recording_id,
            track_id,
        ),
    )


def save_record(conn: sqlite3.Connection, track_id: str, record: TrackRecord) -> None:
    conn.execute(
        "INSERT INTO source_records (source, source_key, track_id, record, imported_at)"
        " VALUES (?, ?, ?, ?, ?) ON CONFLICT (source, source_key)"
        " DO UPDATE SET record = excluded.record, imported_at = excluded.imported_at",
        (record.source, record.source_key, track_id, record.model_dump_json(), _now()),
    )


def load_tracks(conn: sqlite3.Connection) -> list[Track]:
    """All canonical tracks in stable creation order, each with its source records."""
    records: defaultdict[str, list[TrackRecord]] = defaultdict(list)
    for row in conn.execute("SELECT track_id, record FROM source_records ORDER BY rowid"):
        records[row["track_id"]].append(TrackRecord.model_validate_json(row["record"]))
    rows = conn.execute("SELECT * FROM tracks ORDER BY created_at, rowid")
    return [
        Track(
            id=row["id"],
            title=row["title"],
            artists=tuple(json.loads(row["artists"])),
            album=row["album"],
            duration_ms=row["duration_ms"],
            year=row["year"],
            isrc=row["isrc"],
            musicbrainz_recording_id=row["musicbrainz_recording_id"],
            version=row["version"],
            records=tuple(records[row["id"]]),
        )
        for row in rows
    ]


@dataclass(frozen=True)
class SourceRemoval:
    records: int
    tracks_removed: int  # Tracks known only through the source.
    tracks_rebuilt: int  # Tracks whose canonical fields are re-derived from other sources.
    results_removed: int  # Saved results that included a removed track.


def remove_source(conn: sqlite3.Connection, source: str, *, apply: bool) -> SourceRemoval:
    """Delete every record from ``source`` (or, without ``apply``, only count what would go).

    Canonical fields may have been copied from the source, so surviving tracks are rebuilt
    from their remaining records; saved results that include a removed track are deleted.
    """
    affected = [
        row[0]
        for row in conn.execute(
            "SELECT DISTINCT track_id FROM source_records WHERE source = ?", (source,)
        )
    ]
    remaining: defaultdict[str, list[TrackRecord]] = defaultdict(list)
    for row in conn.execute(
        "SELECT track_id, record FROM source_records WHERE source != ? ORDER BY rowid", (source,)
    ):
        remaining[row["track_id"]].append(TrackRecord.model_validate_json(row["record"]))
    orphans = [t for t in affected if not remaining.get(t)]
    marks = ",".join("?" * len(orphans))
    runs = [
        row[0]
        for row in conn.execute(
            f"SELECT DISTINCT run_id FROM assignments WHERE track_id IN ({marks})", orphans
        )
    ]
    records = conn.execute(
        "SELECT COUNT(*) FROM source_records WHERE source = ?", (source,)
    ).fetchone()[0]
    summary = SourceRemoval(records, len(orphans), len(affected) - len(orphans), len(runs))
    if not apply:
        return summary
    with conn:
        conn.executemany("DELETE FROM organization_runs WHERE id = ?", [(r,) for r in runs])
        conn.execute("DELETE FROM source_records WHERE source = ?", (source,))
        conn.execute(f"DELETE FROM enrichment_lookups WHERE track_id IN ({marks})", orphans)
        conn.execute(f"DELETE FROM tracks WHERE id IN ({marks})", orphans)
        for track_id in affected:
            if remaining.get(track_id):
                first, *others = remaining[track_id]
                assignments = ", ".join(f"{c.strip()} = ?" for c in _CANONICAL.split(","))
                conn.execute(
                    f"UPDATE tracks SET {assignments} WHERE id = ?", (*_canonical(first), track_id)
                )
                for record in others:
                    fill_missing_fields(conn, track_id, record)
    return summary
