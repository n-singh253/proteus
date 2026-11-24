"""JSON track-list import and organization-result export (round-trippable)."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, ValidationError, field_validator

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
    name="json",
    capabilities=frozenset({Capability.IMPORT, Capability.EXPORT}),
    description='A list of track objects, {"tracks": [...]}, or a Proteus JSON export.',
)


class JsonTrack(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str | None = None
    title: str
    artists: tuple[str, ...] = ()
    album: str | None = None
    duration_ms: int | None = None
    year: int | None = None
    genres: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    isrc: str | None = None
    musicbrainz_recording_id: str | None = None
    path: str | None = None
    analysis_permitted: bool = True

    @field_validator("artists", "genres", "tags", mode="before")
    @classmethod
    def _accept_delimited(cls, value: Any) -> Any:
        return fields.split_list(value, ";") if isinstance(value, str) else value


def _track_objects(document: Any) -> list[Any]:
    if isinstance(document, list):
        return document
    if isinstance(document, dict):
        if isinstance(document.get("tracks"), list):
            return document["tracks"]
        if isinstance(document.get("playlists"), list):
            return [t for p in document["playlists"] for t in p.get("tracks", [])]
    raise ValueError('expected a list of tracks, {"tracks": [...]}, or a Proteus export')


def read_json(path: Path, *, analysis_permitted: bool = True) -> ImportBatch:
    try:
        document = json.loads(path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path} is not valid JSON: {exc}") from exc
    records, failures, warnings = [], [], []
    for position, obj in enumerate(_track_objects(document)):
        location = f"{path}[{position}]"
        try:
            item = JsonTrack.model_validate(obj)
            local_path = None
            if item.path:
                resolved = fields.resolve_local_path(item.path, path.parent)
                if resolved.is_file():
                    local_path = str(resolved)
                else:
                    warnings.append(ImportIssue(location, f"file not found: {resolved}"))
            key = (
                item.id
                or local_path
                or fields.content_key(item.title, item.artists, item.album, item.duration_ms)
            )
            records.append(
                TrackRecord(
                    source="json",
                    source_key=key,
                    local_path=local_path,
                    analysis_permitted=analysis_permitted and item.analysis_permitted,
                    **item.model_dump(exclude={"id", "path", "analysis_permitted"}),
                )
            )
        except ValidationError as exc:
            failures.append(ImportIssue(location, fields.describe_error(exc)))
    return ImportBatch("json", tuple(records), tuple(failures), tuple(warnings))


def _track_object(track: Track) -> dict[str, Any]:
    permitted = track.analysis_records
    return {
        "id": track.id,
        "title": track.title,
        "artists": list(track.artists),
        "album": track.album,
        "duration_ms": track.duration_ms,
        "year": track.year,
        "genres": list(dict.fromkeys(g for r in permitted for g in r.genres)),
        "tags": list(dict.fromkeys(t for r in permitted for t in r.tags)),
        "isrc": track.isrc,
        "musicbrainz_recording_id": track.musicbrainz_recording_id,
        "path": track.local_path,
        "analysis_permitted": track.exportable_for_analysis,
    }


def write_result(
    result: OrganizationResult, tracks: Mapping[str, Track], out_dir: Path, options: ExportOptions
) -> ExportReport:
    target = out_dir / "organization.json"
    fields.ensure_writable([target], options.overwrite)
    by_track = {a.track_id: a for a in result.assignments}
    document = {
        "format": "proteus-organization",
        "format_version": 1,
        "result": result.model_dump(mode="json", exclude={"playlists", "assignments"}),
        "playlists": [
            {
                "name": p.name,
                "description": p.description,
                "cohesion": p.cohesion,
                "tracks": [
                    _track_object(tracks[t])
                    | {
                        "similarity": by_track[t].similarity,
                        "flags": list(by_track[t].flags),
                    }
                    for t in p.track_ids
                ],
            }
            for p in result.playlists
        ],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(document, indent=2, ensure_ascii=False), encoding="utf-8")
    return ExportReport("json", (str(target),), len(result.assignments))
