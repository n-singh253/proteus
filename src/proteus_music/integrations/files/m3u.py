"""M3U/M3U8 playlist import and export."""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import url2pathname

from proteus_music import _native
from proteus_music.integrations.contract import (
    AdapterInfo,
    Capability,
    ExportOptions,
    ExportReport,
    ImportBatch,
    ImportIssue,
)
from proteus_music.integrations.files import fields
from proteus_music.integrations.local.scanner import audio_file_record
from proteus_music.models.organization import OrganizationResult
from proteus_music.models.track import Track, TrackRecord

ADAPTER = AdapterInfo(
    name="m3u",
    capabilities=frozenset({Capability.IMPORT, Capability.EXPORT}),
    description="M3U/M3U8 playlists referencing local files (exports one .m3u8 per playlist).",
)

_EXTINF = re.compile(r"#EXTINF:\s*(-?\d+(?:\.\d+)?)[^,]*,(.*)")
_URL_SCHEME = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]+://")


@dataclass(frozen=True)
class _Entry:
    location: str
    path: Path
    duration_s: float | None
    display: str | None


def _read_text(path: Path) -> str:
    raw = path.read_bytes()
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        if path.suffix.lower() == ".m3u8":
            raise ValueError(f"{path} is not valid UTF-8 as required for .m3u8") from None
        return raw.decode("latin-1")  # Legacy .m3u files are conventionally Latin-1.


def _parse(path: Path) -> tuple[list[_Entry], list[ImportIssue]]:
    entries, failures = [], []
    pending: tuple[float | None, str | None] = (None, None)
    for number, line in enumerate(_read_text(path).splitlines(), start=1):
        line = line.strip()
        if match := _EXTINF.match(line):
            seconds = float(match.group(1))
            pending = (seconds if seconds > 0 else None, match.group(2).strip() or None)
            continue
        if not line or line.startswith("#"):
            continue
        location = f"{path}:{number}"
        if line.lower().startswith("file://"):
            line = url2pathname(urlparse(line).path)
        elif _URL_SCHEME.match(line):
            failures.append(ImportIssue(location, "remote URLs are not supported"))
            pending = (None, None)
            continue
        entries.append(_Entry(location, fields.resolve_local_path(line, path.parent), *pending))
        pending = (None, None)
    return entries, failures


def _extinf_record(entry: _Entry, analysis_permitted: bool) -> TrackRecord:
    artist, _, title = (entry.display or "").partition(" - ")
    if not title:
        artist, title = "", artist
    duration = round(entry.duration_s * 1000) if entry.duration_s else None
    return TrackRecord(
        source="m3u",
        source_key=fields.content_key(str(entry.path), entry.display),
        title=title,
        artists=(artist,) if artist else (),
        duration_ms=duration,
        analysis_permitted=analysis_permitted,
    )


def read_m3u(path: Path, *, analysis_permitted: bool = True) -> ImportBatch:
    """Import playlist entries, reading tags from referenced files that exist."""
    entries, failures = _parse(path)
    existing: list[_Entry] = []
    missing: list[_Entry] = []
    for entry in entries:
        (existing if entry.path.is_file() else missing).append(entry)
    files, read_failures = _native.read_metadata([e.path for e in existing])
    failures.extend(ImportIssue(f.path, f.reason) for f in read_failures)
    records = [audio_file_record(f, analysis_permitted=analysis_permitted) for f in files]
    warnings = []
    for entry in missing:
        if entry.display:
            records.append(_extinf_record(entry, analysis_permitted))
            warnings.append(
                ImportIssue(entry.location, f"file not found, used #EXTINF: {entry.path}")
            )
        else:
            failures.append(ImportIssue(entry.location, f"file not found: {entry.path}"))
    return ImportBatch("m3u", tuple(records), tuple(failures), tuple(warnings))


def _entry_path(track_path: str, out_dir: Path, relative: bool) -> str:
    if not relative:
        return track_path
    try:
        return os.path.relpath(track_path, out_dir)
    except ValueError:  # Different drives on Windows have no relative path.
        return track_path


def write_result(
    result: OrganizationResult, tracks: Mapping[str, Track], out_dir: Path, options: ExportOptions
) -> ExportReport:
    out_dir = out_dir.resolve()
    targets = [
        out_dir / f"{p.index + 1:02d} {fields.safe_filename(p.name)}.m3u8" for p in result.playlists
    ]
    fields.ensure_writable(targets, options.overwrite)
    out_dir.mkdir(parents=True, exist_ok=True)
    written, exported, unexportable = [], 0, []
    for playlist, target in zip(result.playlists, targets, strict=True):
        lines = ["#EXTM3U", f"#PLAYLIST:{playlist.name}"]
        for track_id in playlist.track_ids:
            track = tracks[track_id]
            if not track.local_path:
                reason = f"no local file for '{track.title}' (playlist '{playlist.name}')"
                unexportable.append(ImportIssue(track_id, reason))
                continue
            seconds = round(track.duration_ms / 1000) if track.duration_ms else -1
            lines.append(f"#EXTINF:{seconds},{', '.join(track.artists)} - {track.title}")
            lines.append(_entry_path(track.local_path, out_dir, options.relative_paths))
            exported += 1
        if (
            len(lines) > 2
        ):  # Skip playlists with no exportable entries rather than write empty files.
            target.write_text("\n".join(lines) + "\n", encoding="utf-8")
            written.append(str(target))
    return ExportReport("m3u8", tuple(written), exported, tuple(unexportable))
