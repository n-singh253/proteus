"""Persistence for external metadata lookups: outcome, provenance and refresh time."""

from __future__ import annotations

import sqlite3
from collections import Counter
from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class LookupRow:
    provider: str
    track_id: str
    query: str
    status: str
    recording_id: str | None
    confidence: float | None
    candidates: int
    detail: str
    license: str
    fetched_at: datetime
    refresh_after: datetime


def refresh_times(conn: sqlite3.Connection, provider: str) -> dict[str, datetime]:
    rows = conn.execute(
        "SELECT track_id, refresh_after FROM enrichment_lookups WHERE provider = ?", (provider,)
    )
    return {r["track_id"]: datetime.fromisoformat(r["refresh_after"]) for r in rows}


def save_lookup(conn: sqlite3.Connection, row: LookupRow) -> None:
    conn.execute(
        "INSERT INTO enrichment_lookups (provider, track_id, query, status, recording_id,"
        " confidence, candidates, detail, license, fetched_at, refresh_after)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
        " ON CONFLICT (provider, track_id) DO UPDATE SET query = excluded.query,"
        " status = excluded.status, recording_id = excluded.recording_id,"
        " confidence = excluded.confidence, candidates = excluded.candidates,"
        " detail = excluded.detail, license = excluded.license,"
        " fetched_at = excluded.fetched_at, refresh_after = excluded.refresh_after",
        (
            row.provider,
            row.track_id,
            row.query,
            row.status,
            row.recording_id,
            row.confidence,
            row.candidates,
            row.detail,
            row.license,
            row.fetched_at.isoformat(),
            row.refresh_after.isoformat(),
        ),
    )


def status_counts(conn: sqlite3.Connection, provider: str) -> Counter[str]:
    rows = conn.execute(
        "SELECT status, COUNT(*) AS n FROM enrichment_lookups WHERE provider = ? GROUP BY status",
        (provider,),
    )
    return Counter({r["status"]: r["n"] for r in rows})
