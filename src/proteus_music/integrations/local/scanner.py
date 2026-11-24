"""Local audio collection adapter backed by the native scanner."""

from __future__ import annotations

from pathlib import Path

from proteus_music import _native
from proteus_music.integrations.contract import AdapterInfo, Capability, ImportBatch, ImportIssue
from proteus_music.integrations.files.fields import split_genres
from proteus_music.models.track import TrackRecord

ADAPTER = AdapterInfo(
    name="local",
    capabilities=frozenset({Capability.IMPORT}),
    description="Audio files (MP3, FLAC, WAV, AAC, M4A, Ogg Vorbis, Opus), detected by content.",
)


def audio_file_record(file: _native.AudioFile, *, analysis_permitted: bool = True) -> TrackRecord:
    artists = tuple(a.strip() for value in file.artists for a in value.split(";"))
    return TrackRecord(
        source="local",
        source_key=file.path,
        title=file.title or Path(file.path).stem,
        artists=artists or ((file.album_artist,) if file.album_artist else ()),
        album=file.album,
        duration_ms=file.duration_ms,
        year=file.year if file.year and file.year >= 1000 else None,
        genres=split_genres(file.genres),
        isrc=file.isrc,
        musicbrainz_recording_id=file.musicbrainz_recording_id,
        local_path=file.path,
        analysis_permitted=analysis_permitted,
    )


def read_directory(
    root: Path, *, recursive: bool = True, max_threads: int | None = None
) -> ImportBatch:
    files, failures = _native.scan_library(root.expanduser().resolve(), recursive, max_threads)
    return ImportBatch(
        source="local",
        records=tuple(audio_file_record(f) for f in files),
        failures=tuple(ImportIssue(f.path, f.reason) for f in failures),
    )
