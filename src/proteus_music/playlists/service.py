"""Organization and export services shared by the CLI and API."""

from __future__ import annotations

import sqlite3
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from proteus_music.engine.evaluation.validation import validate_result
from proteus_music.engine.features.metadata import describe
from proteus_music.engine.pipeline import PlaylistPlan, plan_playlists
from proteus_music.integrations.contract import AdapterInfo, ExportOptions, ExportReport
from proteus_music.integrations.files import csv_format, json_format, m3u
from proteus_music.integrations.local import scanner
from proteus_music.library.audio import load_acoustic, permitted_audio
from proteus_music.models.organization import FeatureSet, OrganizationConfig, OrganizationResult
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


def new_result(
    plan: PlaylistPlan, config: OrganizationConfig, track_ids: tuple[str, ...], notes: tuple = ()
) -> OrganizationResult:
    return OrganizationResult(
        id=uuid.uuid4().hex[:12],
        created_at=datetime.now(UTC),
        config=config,
        feature_version=plan.feature_version,
        snapshot_track_ids=track_ids,
        playlists=plan.playlists,
        assignments=plan.assignments,
        warnings=(*notes, *plan.warnings),
        quality=plan.quality,
        selection=plan.selection,
        categories=plan.categories,
    )


def organize_library(
    conn: sqlite3.Connection,
    config: OrganizationConfig,
    checkpoint: Callable[[str], None] | None = None,
) -> OrganizationResult:
    """Organize every canonical track into exclusive playlists, validate, and save the result.

    ``checkpoint`` is called between stages and may raise to cancel; the result is written in
    one transaction after the last checkpoint, so a cancelled or failed run saves nothing.
    """
    checkpoint = checkpoint or (lambda stage: None)
    checkpoint("loading library")
    tracks = load_tracks(conn)
    restricted = sum(1 for t in tracks if not t.analysis_records)
    notes: tuple[str, ...] = ()
    if restricted:
        notes = (
            f"{restricted} identification-only track(s) are assigned but contribute no "
            "analysis features",
        )
    acoustic = {} if config.features is FeatureSet.METADATA else load_acoustic(conn, tracks)
    files = len(permitted_audio(tracks))
    if config.features is FeatureSet.AUDIO or (config.features is FeatureSet.FUSED and files):
        notes = (
            *notes,
            f"{len(acoustic)} of {len(tracks)} track(s) have analyzed audio ({files} with a "
            "permitted local file); run `proteus analyze audio` to analyze new files",
        )
    checkpoint("organizing")
    plan = plan_playlists([describe(t, acoustic.get(t.id)) for t in tracks], config)
    checkpoint("validating")
    result = new_result(plan, config, tuple(t.id for t in tracks), notes)
    validate_result(result)
    checkpoint("saving")
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
    validate_result(result)
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
    tracks = {t.id: t for t in load_tracks(conn)}
    return EXPORTERS[target](result, tracks, out_dir.expanduser(), options)
