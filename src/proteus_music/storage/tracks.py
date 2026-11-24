"""Persistence for canonical tracks and their source records."""

from __future__ import annotations

import json
import sqlite3
from collections import defaultdict
from datetime import UTC, datetime

from proteus_music.library.identity import parse_title
from proteus_music.models.track import Track, TrackRecord


def _now() -> str:
    return datetime.now(UTC).isoformat()


def record_track_id(conn: sqlite3.Connection, source: str, source_key: str) -> str | None:
    row = conn.execute(
        "SELECT track_id FROM source_records WHERE source = ? AND source_key = ?",
        (source, source_key),
    ).fetchone()
    return row["track_id"] if row else None


def insert_track(conn: sqlite3.Connection, track_id: str, record: TrackRecord) -> None:
    parts = parse_title(record.title)
    artists = list(dict.fromkeys([*record.artists, *parts.featured]))
    conn.execute(
        "INSERT INTO tracks (id, title, artists, album, duration_ms, year, isrc,"
        " musicbrainz_recording_id, version, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            track_id,
            record.title,
            json.dumps(artists),
            record.album,
            record.duration_ms,
            record.year,
            record.isrc,
            record.musicbrainz_recording_id,
            parts.version,
            _now(),
        ),
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
