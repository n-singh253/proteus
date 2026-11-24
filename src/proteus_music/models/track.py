"""Normalized, provider-independent track model."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator


class TrackRecord(BaseModel):
    """One track as seen by one source, normalized at ingestion.

    ``analysis_permitted`` records whether this source's metadata may feed analysis and
    organization; records without permission are still usable for identity and export.
    """

    model_config = ConfigDict(frozen=True)

    source: str
    source_key: str
    title: str
    artists: tuple[str, ...] = ()
    album: str | None = None
    duration_ms: int | None = Field(default=None, gt=0)
    year: int | None = Field(default=None, ge=1000, le=9999)
    genres: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    isrc: str | None = None
    musicbrainz_recording_id: str | None = None
    local_path: str | None = None
    analysis_permitted: bool = True

    @field_validator("title")
    @classmethod
    def _title_required(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("title must not be empty")
        return value

    @field_validator("artists", "genres", "tags")
    @classmethod
    def _clean_list(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        cleaned = (v.strip() for v in values)
        return tuple(dict.fromkeys(v for v in cleaned if v))

    @field_validator("isrc")
    @classmethod
    def _normalize_isrc(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.replace("-", "").strip().upper()
        return value or None


class Track(BaseModel):
    """A canonical recording with every source record that resolved to it."""

    model_config = ConfigDict(frozen=True)

    id: str
    title: str
    artists: tuple[str, ...]
    album: str | None
    duration_ms: int | None
    year: int | None
    isrc: str | None
    musicbrainz_recording_id: str | None
    version: str | None
    records: tuple[TrackRecord, ...]

    @property
    def local_path(self) -> str | None:
        return next((r.local_path for r in self.records if r.local_path), None)

    @property
    def analysis_records(self) -> tuple[TrackRecord, ...]:
        return tuple(r for r in self.records if r.analysis_permitted)
