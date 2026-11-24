"""Organization and export services shared by the CLI and API."""

from __future__ import annotations

import sqlite3
import uuid
from datetime import UTC, datetime
from pathlib import Path

from proteus_music.engine.evaluation.validation import validate_result
from proteus_music.engine.features.metadata import describe
from proteus_music.engine.pipeline import plan_playlists
from proteus_music.integrations.contract import AdapterInfo, ExportOptions, ExportReport
from proteus_music.integrations.files import csv_format, json_format, m3u
from proteus_music.integrations.local import scanner
from proteus_music.models.organization import OrganizationConfig, OrganizationResult
from proteus_music.storage import results as result_store
from proteus_music.storage.tracks import load_tracks

EXPORTERS = {
    "m3u8": m3u.write_result,
    "csv": csv_format.write_result,
    "json": json_format.write_result,
}


ADAPTERS: tuple[AdapterInfo, ...] = (
    scanner.ADAPTER,
    csv_format.ADAPTER,
    json_format.ADAPTER,
    m3u.ADAPTER,
)


def organize_library(conn: sqlite3.Connection, config: OrganizationConfig) -> OrganizationResult:
    """Organize every canonical track into exclusive playlists, validate, and save the result."""
    tracks = load_tracks(conn)
    plan = plan_playlists([describe(t) for t in tracks], config)
    result = OrganizationResult(
        id=uuid.uuid4().hex[:12],
        created_at=datetime.now(UTC),
        config=config,
        feature_version=plan.feature_version,
        snapshot_track_ids=tuple(t.id for t in tracks),
        playlists=plan.playlists,
        assignments=plan.assignments,
        warnings=plan.warnings,
    )
    validate_result(result)
    result_store.save_result(conn, result)
    return result


def get_result(conn: sqlite3.Connection, result_id: str | None = None) -> OrganizationResult:
    """Load a saved result (the latest when no ID is given)."""
    result_id = result_id or result_store.latest_result_id(conn)
    if result_id is None:
        raise LookupError("no organization results yet; run `proteus organize` first")
    result = result_store.load_result(conn, result_id)
    if result is None:
        raise LookupError(f"no organization result with ID {result_id!r}")
    return result


def export_result(
    conn: sqlite3.Connection,
    target: str,
    out_dir: Path,
    *,
    result_id: str | None = None,
    options: ExportOptions = ExportOptions(),
) -> ExportReport:
    if target not in EXPORTERS:
        raise ValueError(f"unknown export target {target!r}; available: {', '.join(EXPORTERS)}")
    result = get_result(conn, result_id)
    validate_result(result)
    tracks = {t.id: t for t in load_tracks(conn)}
    return EXPORTERS[target](result, tracks, out_dir.expanduser(), options)
