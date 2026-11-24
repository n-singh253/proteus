"""Organization configuration and reproducible results."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class OrganizationConfig(BaseModel):
    """Hard constraints and settings for one organization run."""

    model_config = ConfigDict(frozen=True)

    playlists: int = Field(ge=1)
    min_size: int = Field(default=1, ge=1)
    max_size: int | None = Field(default=None, ge=1)
    algorithm: str = "spherical-kmeans"
    seed: int = 0

    @model_validator(mode="after")
    def _sizes_consistent(self) -> OrganizationConfig:
        if self.max_size is not None and self.max_size < self.min_size:
            raise ValueError(f"max_size ({self.max_size}) is below min_size ({self.min_size})")
        return self


class AssignmentFlag(StrEnum):
    MISSING_FEATURES = "missing-features"
    LOW_SIMILARITY = "low-similarity"
    AMBIGUOUS = "ambiguous"
    CONSTRAINT_DISPLACED = "constraint-displaced"


class Assignment(BaseModel):
    model_config = ConfigDict(frozen=True)

    track_id: str
    playlist_index: int
    similarity: float
    margin: float
    flags: tuple[AssignmentFlag, ...] = ()


class Playlist(BaseModel):
    model_config = ConfigDict(frozen=True)

    index: int
    name: str
    description: str
    track_ids: tuple[str, ...]
    cohesion: float


class OrganizationResult(BaseModel):
    """A complete, exclusive assignment of a track snapshot to playlists."""

    model_config = ConfigDict(frozen=True)

    id: str
    created_at: datetime
    config: OrganizationConfig
    feature_version: str
    snapshot_track_ids: tuple[str, ...]
    playlists: tuple[Playlist, ...]
    assignments: tuple[Assignment, ...]
    warnings: tuple[str, ...] = ()

    @property
    def low_confidence(self) -> tuple[Assignment, ...]:
        return tuple(a for a in self.assignments if a.flags)
