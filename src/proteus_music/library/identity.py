"""Deterministic recording identity resolution.

Records merge only on a shared ISRC, a shared MusicBrainz recording ID, or an exact match of
normalized title, primary artist and version descriptor with compatible durations. Distinct
versions (live, remix, remaster, ...) never merge, even when a shared identifier says otherwise:
such identifier/version conflicts stay separate and are reported for review.
"""

from __future__ import annotations

import re
import unicodedata
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass

from proteus_music.models.track import Track, TrackRecord

DURATION_TOLERANCE_MS = 3000

_VERSION_WORDS = re.compile(
    r"\b(live|acoustic|remix(ed)?|mix|instrumental|remaster(ed)?|edit|extended|clean|explicit|"
    r"demo|unplugged|version|mono|stereo|rework|dub|karaoke|a ?cappella|sped up|slowed|session)\b",
    re.IGNORECASE,
)
_FEATURING = re.compile(
    r"\s*[(\[]?\s*\b(?:feat\.?|ft\.|featuring)\s+(.+?)(?:[)\]]|(?=\s+-\s+)|$)", re.IGNORECASE
)
_BRACKETED = re.compile(r"\s*[(\[]([^)\]]*)[)\]]")
_DASH_SUFFIX = re.compile(r"\s+-\s+([^-]+)$")


def normalize_text(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).casefold().replace("&", " and ")
    value = "".join(ch if ch.isalnum() else " " for ch in value)
    return " ".join(value.split())


@dataclass(frozen=True)
class TitleParts:
    base: str
    version: str | None
    featured: tuple[str, ...]


def parse_title(title: str) -> TitleParts:
    featured: list[str] = []
    for match in _FEATURING.finditer(title):
        featured.extend(a.strip() for a in re.split(r",|&| and ", match.group(1)) if a.strip())
    remaining = _FEATURING.sub("", title)

    versions: list[str] = []

    def take_version(match: re.Match[str]) -> str:
        text = match.group(1)
        if _VERSION_WORDS.search(text):
            versions.append(normalize_text(text))
            return ""
        return match.group(0)

    remaining = _BRACKETED.sub(take_version, remaining)
    remaining = _DASH_SUFFIX.sub(take_version, remaining)
    base = remaining.strip() or title.strip()
    return TitleParts(base, "; ".join(versions) or None, tuple(featured))


def identity_key(record: TrackRecord) -> tuple[str, str, str]:
    parts = parse_title(record.title)
    artist = normalize_text(record.artists[0]) if record.artists else ""
    return normalize_text(parts.base), artist, parts.version or ""


@dataclass(frozen=True)
class _Entry:
    track_id: str
    isrc: str | None
    mbid: str | None
    duration_ms: int | None
    version: str


def _version(record: TrackRecord) -> str:
    return identity_key(record)[2]


def _compatible(entry: _Entry, record: TrackRecord) -> bool:
    if entry.isrc and record.isrc and entry.isrc != record.isrc:
        return False
    mbid = record.musicbrainz_recording_id
    if entry.mbid and mbid and entry.mbid != mbid:
        return False
    if entry.duration_ms is None and record.duration_ms is None:
        return True
    if entry.duration_ms is None or record.duration_ms is None:
        return False
    return abs(entry.duration_ms - record.duration_ms) <= DURATION_TOLERANCE_MS


class IdentityIndex:
    """In-memory lookup from identifying signals to canonical track IDs."""

    def __init__(self) -> None:
        self._by_isrc: dict[str, _Entry] = {}
        self._by_mbid: dict[str, _Entry] = {}
        self._by_key: defaultdict[tuple[str, str, str], list[_Entry]] = defaultdict(list)
        self.conflicts: list[str] = []

    @classmethod
    def from_tracks(cls, tracks: Iterable[Track]) -> IdentityIndex:
        index = cls()
        for track in tracks:
            for record in track.records:
                index.add(track.id, record)
        return index

    def resolve(self, record: TrackRecord) -> str | None:
        """Matching canonical track ID, or None (identifier/version conflicts never merge)."""
        version = _version(record)
        for label, value, table in (
            ("ISRC", record.isrc, self._by_isrc),
            ("MusicBrainz ID", record.musicbrainz_recording_id, self._by_mbid),
        ):
            entry = table.get(value) if value else None
            if entry is None:
                continue
            if entry.version == version:
                return entry.track_id
            self.conflicts.append(
                f"'{record.title}' shares {label} {value} with track {entry.track_id} but has "
                f"version '{version or 'original'}' vs '{entry.version or 'original'}'; "
                "kept separate"
            )
            return None
        for entry in self._by_key.get(identity_key(record), ()):
            if _compatible(entry, record):
                return entry.track_id
        return None

    def add(self, track_id: str, record: TrackRecord) -> None:
        key = identity_key(record)
        entry = _Entry(
            track_id, record.isrc, record.musicbrainz_recording_id, record.duration_ms, key[2]
        )
        if record.isrc:
            self._by_isrc.setdefault(record.isrc, entry)
        if record.musicbrainz_recording_id:
            self._by_mbid.setdefault(record.musicbrainz_recording_id, entry)
        self._by_key[key].append(entry)
