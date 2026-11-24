"""Persistence for organization results."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime

from proteus_music.models.organization import (
    Assignment,
    AssignmentFlag,
    OrganizationConfig,
    OrganizationResult,
    Playlist,
)


def save_result(conn: sqlite3.Connection, result: OrganizationResult) -> None:
    with conn:
        conn.execute(
            "INSERT INTO organization_runs (id, created_at, config, feature_version, snapshot,"
            " warnings) VALUES (?, ?, ?, ?, ?, ?)",
            (
                result.id,
                result.created_at.isoformat(),
                result.config.model_dump_json(),
                result.feature_version,
                json.dumps(result.snapshot_track_ids),
                json.dumps(result.warnings),
            ),
        )
        conn.executemany(
            "INSERT INTO playlists (run_id, idx, name, description, cohesion)"
            " VALUES (?, ?, ?, ?, ?)",
            [(result.id, p.index, p.name, p.description, p.cohesion) for p in result.playlists],
        )
        by_track = {a.track_id: a for a in result.assignments}
        conn.executemany(
            "INSERT INTO assignments (run_id, track_id, playlist_idx, position, similarity, margin,"
            " flags) VALUES (?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    result.id,
                    t,
                    p.index,
                    pos,
                    by_track[t].similarity,
                    by_track[t].margin,
                    json.dumps(by_track[t].flags),
                )
                for p in result.playlists
                for pos, t in enumerate(p.track_ids)
            ],
        )


def latest_result_id(conn: sqlite3.Connection) -> str | None:
    row = conn.execute(
        "SELECT id FROM organization_runs ORDER BY created_at DESC, rowid DESC LIMIT 1"
    ).fetchone()
    return row["id"] if row else None


def list_results(conn: sqlite3.Connection) -> list[tuple[str, datetime, int, int]]:
    """(id, created_at, playlist count, track count) for every saved result, newest first."""
    rows = conn.execute(
        "SELECT r.id, r.created_at, json_array_length(r.snapshot) AS tracks,"
        " (SELECT COUNT(*) FROM playlists p WHERE p.run_id = r.id) AS playlists"
        " FROM organization_runs r ORDER BY r.created_at DESC, r.rowid DESC"
    )
    return [
        (r["id"], datetime.fromisoformat(r["created_at"]), r["playlists"], r["tracks"])
        for r in rows
    ]


def load_result(conn: sqlite3.Connection, result_id: str) -> OrganizationResult | None:
    run = conn.execute("SELECT * FROM organization_runs WHERE id = ?", (result_id,)).fetchone()
    if run is None:
        return None
    rows = conn.execute(
        "SELECT * FROM assignments WHERE run_id = ? ORDER BY playlist_idx, position", (result_id,)
    ).fetchall()
    members: dict[int, list[str]] = {}
    for row in rows:
        members.setdefault(row["playlist_idx"], []).append(row["track_id"])
    playlists = tuple(
        Playlist(
            index=p["idx"],
            name=p["name"],
            description=p["description"],
            track_ids=tuple(members.get(p["idx"], ())),
            cohesion=p["cohesion"],
        )
        for p in conn.execute("SELECT * FROM playlists WHERE run_id = ? ORDER BY idx", (result_id,))
    )
    assignments = tuple(
        Assignment(
            track_id=row["track_id"],
            playlist_index=row["playlist_idx"],
            similarity=row["similarity"],
            margin=row["margin"],
            flags=tuple(AssignmentFlag(f) for f in json.loads(row["flags"])),
        )
        for row in rows
    )
    return OrganizationResult(
        id=run["id"],
        created_at=datetime.fromisoformat(run["created_at"]),
        config=OrganizationConfig.model_validate_json(run["config"]),
        feature_version=run["feature_version"],
        snapshot_track_ids=tuple(json.loads(run["snapshot"])),
        playlists=playlists,
        assignments=assignments,
        warnings=tuple(json.loads(run["warnings"])),
    )
