"""CSV library import with configurable column mapping, and CSV result export."""

from __future__ import annotations

import csv
import io
from collections.abc import Mapping
from pathlib import Path

from pydantic import ValidationError

from proteus_music.integrations.contract import (
    AdapterInfo,
    Capability,
    ExportOptions,
    ExportReport,
    ImportBatch,
    ImportIssue,
)
from proteus_music.integrations.files import fields
from proteus_music.models.organization import OrganizationResult
from proteus_music.models.track import Track, TrackRecord

ADAPTER = AdapterInfo(
    name="csv",
    capabilities=frozenset({Capability.IMPORT, Capability.EXPORT}),
    description="Delimited track lists with configurable column mapping.",
)

# Recognized header names (case-insensitive) for each field; override with an explicit mapping.
DEFAULT_COLUMNS: dict[str, tuple[str, ...]] = {
    "id": ("id", "track id", "track_id", "uri", "track uri"),
    "title": ("title", "track name", "track", "name", "song"),
    "artists": ("artists", "artist", "artist name", "artist name(s)", "artist names"),
    "album": ("album", "album name"),
    "duration_ms": ("duration_ms", "duration (ms)", "track duration (ms)"),
    "duration": ("duration", "length", "time"),
    "year": ("year", "release year"),
    "release_date": ("release date", "release_date", "album release date", "date"),
    "genres": ("genres", "genre"),
    "tags": ("tags", "tag", "moods", "mood"),
    "isrc": ("isrc",),
    "musicbrainz_recording_id": ("musicbrainz_recording_id", "musicbrainz recording id", "mbid"),
    "path": ("path", "location", "file", "filename"),
    "analysis_permitted": ("analysis_permitted",),
}
_BOOLEANS = {"true": True, "1": True, "yes": True, "false": False, "0": False, "no": False}

EXPORT_COLUMNS = (
    "playlist", "position", "title", "artists", "album", "duration_ms", "year", "isrc",
    "path", "similarity", "flags", "analysis_permitted",
)  # fmt: skip


def resolve_columns(header: list[str], overrides: Mapping[str, str]) -> dict[str, str]:
    unknown = set(overrides) - set(DEFAULT_COLUMNS)
    if unknown:
        raise ValueError(f"unknown field(s) in column mapping: {', '.join(sorted(unknown))}")
    by_lower = {h.strip().lower(): h for h in header}
    columns: dict[str, str] = {}
    for field, aliases in DEFAULT_COLUMNS.items():
        if field in overrides:
            if overrides[field] not in header:
                raise ValueError(f"mapped column {overrides[field]!r} not found in CSV header")
            columns[field] = overrides[field]
        elif match := next((by_lower[a] for a in aliases if a in by_lower), None):
            columns[field] = match
    if "title" not in columns:
        raise ValueError("no title column found; map one explicitly (e.g. title=Song)")
    return columns


def _row_record(
    row: dict[str, str], columns: dict[str, str], base_dir: Path, separator: str, permitted: bool
) -> tuple[TrackRecord, str | None]:
    def get(field: str) -> str | None:
        value = row.get(columns[field]) if field in columns else None
        return value.strip() if value and value.strip() else None

    duration = fields.parse_duration_ms(get("duration_ms"), milliseconds=True)
    if duration is None:
        duration = fields.parse_duration_ms(get("duration"), milliseconds=False)
    warning = None
    local_path = None
    if raw_path := get("path"):
        resolved = fields.resolve_local_path(raw_path, base_dir)
        if resolved.is_file():
            local_path = str(resolved)
        else:
            warning = f"file not found, imported without local path: {resolved}"
    title, artists = get("title") or "", fields.split_list(get("artists"), separator)
    flag = (get("analysis_permitted") or "true").lower()
    if flag not in _BOOLEANS:
        raise ValueError(f"analysis_permitted must be true or false, got {flag!r}")
    record = TrackRecord(
        source="csv",
        source_key=get("id")
        or local_path
        or fields.content_key(title, artists, get("album"), duration),
        title=title,
        artists=artists,
        album=get("album"),
        duration_ms=duration,
        year=fields.parse_year(get("year")) or fields.parse_year(get("release_date")),
        genres=fields.split_genres(fields.split_list(get("genres"), separator)),
        tags=fields.split_list(get("tags"), separator),
        isrc=get("isrc"),
        musicbrainz_recording_id=get("musicbrainz_recording_id"),
        local_path=local_path,
        analysis_permitted=permitted and _BOOLEANS[flag],
    )
    return record, warning


def read_csv(
    path: Path,
    *,
    column_overrides: Mapping[str, str] | None = None,
    separator: str = ";",
    encoding: str = "utf-8-sig",
    analysis_permitted: bool = True,
) -> ImportBatch:
    """Read a CSV track list. ``separator`` splits multi-value cells such as artists and genres."""
    try:
        text = path.read_text(encoding=encoding)
    except UnicodeDecodeError as exc:
        raise ValueError(f"{path} is not valid {encoding}; pass the file's encoding") from exc
    try:
        dialect: type[csv.Dialect] | csv.Dialect = csv.Sniffer().sniff(text[:8192], ",;\t")
    except csv.Error:
        dialect = csv.excel
    reader = csv.DictReader(io.StringIO(text, newline=""), dialect=dialect)
    columns = resolve_columns(list(reader.fieldnames or []), column_overrides or {})
    records, failures, warnings = [], [], []
    for line, row in enumerate(reader, start=2):
        location = f"{path}:{line}"
        try:
            record, warning = _row_record(row, columns, path.parent, separator, analysis_permitted)
        except (ValidationError, ValueError) as exc:
            failures.append(ImportIssue(location, fields.describe_error(exc)))
            continue
        records.append(record)
        if warning:
            warnings.append(ImportIssue(location, warning))
    return ImportBatch("csv", tuple(records), tuple(failures), tuple(warnings))


def write_result(
    result: OrganizationResult, tracks: Mapping[str, Track], out_dir: Path, options: ExportOptions
) -> ExportReport:
    target = out_dir / "playlists.csv"
    fields.ensure_writable([target], options.overwrite)
    out_dir.mkdir(parents=True, exist_ok=True)
    by_track = {a.track_id: a for a in result.assignments}
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(EXPORT_COLUMNS)
        for playlist in result.playlists:
            for position, track_id in enumerate(playlist.track_ids, start=1):
                track, assignment = tracks[track_id], by_track[track_id]
                writer.writerow([
                    playlist.name, position, track.title, "; ".join(track.artists),
                    track.album or "", track.duration_ms or "", track.year or "",
                    track.isrc or "", track.local_path or "", f"{assignment.similarity:.4f}",
                    "; ".join(assignment.flags), str(track.exportable_for_analysis).lower(),
                ])  # fmt: skip
    return ExportReport("csv", (str(target),), len(result.assignments))
