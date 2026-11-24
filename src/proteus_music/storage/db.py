"""SQLite connection management and schema migrations."""

from __future__ import annotations

import sqlite3
from pathlib import Path

# Append-only: each entry upgrades the schema by one version inside a transaction.
MIGRATIONS: tuple[str, ...] = (
    """
    CREATE TABLE tracks (
        id TEXT PRIMARY KEY,
        title TEXT NOT NULL,
        artists TEXT NOT NULL,
        album TEXT,
        duration_ms INTEGER,
        year INTEGER,
        isrc TEXT,
        musicbrainz_recording_id TEXT,
        version TEXT,
        created_at TEXT NOT NULL
    );
    CREATE TABLE source_records (
        source TEXT NOT NULL,
        source_key TEXT NOT NULL,
        track_id TEXT NOT NULL REFERENCES tracks(id),
        record TEXT NOT NULL,
        imported_at TEXT NOT NULL,
        PRIMARY KEY (source, source_key)
    );
    CREATE INDEX source_records_track ON source_records(track_id);
    CREATE TABLE organization_runs (
        id TEXT PRIMARY KEY,
        created_at TEXT NOT NULL,
        config TEXT NOT NULL,
        feature_version TEXT NOT NULL,
        snapshot TEXT NOT NULL,
        warnings TEXT NOT NULL
    );
    CREATE TABLE playlists (
        run_id TEXT NOT NULL REFERENCES organization_runs(id) ON DELETE CASCADE,
        idx INTEGER NOT NULL,
        name TEXT NOT NULL,
        description TEXT NOT NULL,
        cohesion REAL NOT NULL,
        PRIMARY KEY (run_id, idx)
    );
    CREATE TABLE assignments (
        run_id TEXT NOT NULL,
        track_id TEXT NOT NULL REFERENCES tracks(id),
        playlist_idx INTEGER NOT NULL,
        position INTEGER NOT NULL,
        similarity REAL NOT NULL,
        margin REAL NOT NULL,
        flags TEXT NOT NULL,
        PRIMARY KEY (run_id, track_id),
        FOREIGN KEY (run_id, playlist_idx) REFERENCES playlists(run_id, idx) ON DELETE CASCADE
    );
    """,
    """
    ALTER TABLE organization_runs ADD COLUMN details TEXT NOT NULL DEFAULT '{}';
    ALTER TABLE playlists ADD COLUMN origin TEXT NOT NULL DEFAULT 'discovered';
    """,
    """
    CREATE TABLE enrichment_lookups (
        provider TEXT NOT NULL,
        track_id TEXT NOT NULL REFERENCES tracks(id),
        query TEXT NOT NULL,
        status TEXT NOT NULL,
        recording_id TEXT,
        confidence REAL,
        candidates INTEGER NOT NULL,
        detail TEXT NOT NULL,
        license TEXT NOT NULL,
        fetched_at TEXT NOT NULL,
        refresh_after TEXT NOT NULL,
        PRIMARY KEY (provider, track_id)
    );
    """,
)


def connect(path: Path) -> sqlite3.Connection:
    """Open (creating if needed) the Proteus database and bring its schema up to date."""
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    _migrate(conn)
    return conn


def _migrate(conn: sqlite3.Connection) -> None:
    current = conn.execute("PRAGMA user_version").fetchone()[0]
    if current > len(MIGRATIONS):
        raise RuntimeError(
            f"database schema version {current} is newer than this Proteus build supports "
            f"({len(MIGRATIONS)}); upgrade Proteus"
        )
    for version in range(current + 1, len(MIGRATIONS) + 1):
        try:
            conn.executescript(
                f"BEGIN;\n{MIGRATIONS[version - 1]}\nPRAGMA user_version = {version};\nCOMMIT;"
            )
        except sqlite3.Error:
            if conn.in_transaction:
                conn.rollback()
            raise
