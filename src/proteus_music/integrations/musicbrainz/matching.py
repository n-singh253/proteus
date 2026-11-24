"""High-confidence matching of canonical tracks to MusicBrainz recordings.

Only CC0 core fields are read (recording ID, title, disambiguation, length, artist credit,
ISRCs, release dates). Tags and genre associations are supplementary data under
CC BY-NC-SA and are deliberately ignored.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from proteus_music.library.identity import (
    DURATION_TOLERANCE_MS,
    normalize_text,
    parse_title,
    version_kinds,
)
from proteus_music.models.track import Track, TrackRecord

PROVIDER = "musicbrainz"
LICENSE = "CC0-1.0 (MusicBrainz core data)"


class _Credit(BaseModel):
    name: str


class _Release(BaseModel):
    date: str | None = None


class Recording(BaseModel):
    """The permitted subset of a MusicBrainz recording, validated at ingestion."""

    model_config = ConfigDict(extra="ignore", populate_by_name=True, frozen=True)

    id: str
    title: str
    length: int | None = Field(default=None, gt=0)  # Milliseconds.
    disambiguation: str = ""
    artist_credit: tuple[_Credit, ...] = Field(default=(), alias="artist-credit")
    releases: tuple[_Release, ...] = ()
    isrcs: tuple[str, ...] = ()

    @property
    def artists(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(c.name for c in self.artist_credit))

    @property
    def year(self) -> int | None:
        """Earliest release year among the (at most 25) releases MusicBrainz returns."""
        years = [int(r.date[:4]) for r in self.releases if r.date and r.date[:4].isdigit()]
        return min(years) if years else None


class MatchStatus(StrEnum):
    MATCHED = "matched"
    AMBIGUOUS = "ambiguous"
    NO_MATCH = "no-match"
    NOT_FOUND = "not-found"
    CONFLICT = "conflict"
    ERROR = "error"


@dataclass(frozen=True)
class MatchOutcome:
    status: MatchStatus
    recording: Recording | None
    confidence: float | None
    candidates: int
    detail: str


def parse_recordings(objects: list[dict[str, Any]]) -> list[Recording]:
    parsed = []
    for obj in objects:
        try:
            parsed.append(Recording.model_validate(obj))
        except ValidationError:
            continue  # A malformed entry can never be a confident match.
    return parsed


def _consistency(track: Track, recording: Recording) -> tuple[bool, float, str]:
    """(consistent, confidence, reason). Title and version must agree; artist and duration
    must agree when both sides have them, and each verified signal adds confidence."""
    local, remote = parse_title(track.title), parse_title(recording.title)
    if normalize_text(local.base) != normalize_text(remote.base):
        return False, 0.0, "title differs"
    remote_version = version_kinds(f"{recording.title} {recording.disambiguation}")
    if version_kinds(track.version or "") != remote_version:
        return False, 0.0, "different recording version"
    confidence = 0.9
    if track.artists:
        names = {normalize_text(a) for a in recording.artists}
        if normalize_text(track.artists[0]) not in names:
            return False, 0.0, "artist differs"
        confidence += 0.05
    if track.duration_ms and recording.length:
        if abs(track.duration_ms - recording.length) > DURATION_TOLERANCE_MS:
            return False, 0.0, "duration differs"
        confidence += 0.05
    return True, round(confidence, 2), ""


def match(track: Track, recordings: list[Recording] | None, *, by_mbid: bool) -> MatchOutcome:
    if recordings is None or not recordings:
        return MatchOutcome(MatchStatus.NOT_FOUND, None, None, 0, "identifier unknown")
    checked = [(r, *_consistency(track, r)) for r in recordings]
    accepted = [(r, c) for r, ok, c, _ in checked if ok]
    if by_mbid and not accepted:
        reason = checked[0][3]
        return MatchOutcome(MatchStatus.CONFLICT, None, None, 1, f"MBID record: {reason}")
    if not accepted:
        reasons = sorted({reason for *_, reason in checked})
        detail = f"{len(recordings)} candidate(s) rejected: {', '.join(reasons)}"
        return MatchOutcome(MatchStatus.NO_MATCH, None, None, len(recordings), detail)
    if len({r.id for r, _ in accepted}) > 1:
        detail = f"{len(accepted)} consistent recordings share the identifier"
        return MatchOutcome(MatchStatus.AMBIGUOUS, None, None, len(recordings), detail)
    recording, confidence = accepted[0]
    if by_mbid:
        confidence = 1.0
    return MatchOutcome(MatchStatus.MATCHED, recording, confidence, len(recordings), "")


def to_record(recording: Recording) -> TrackRecord:
    """A separate, independently licensed source record; it never alters other records."""
    return TrackRecord(
        source=PROVIDER,
        source_key=recording.id,
        title=recording.title,
        artists=recording.artists,
        duration_ms=recording.length,
        year=recording.year,
        isrc=recording.isrcs[0] if recording.isrcs else None,
        musicbrainz_recording_id=recording.id,
        analysis_permitted=True,
    )
