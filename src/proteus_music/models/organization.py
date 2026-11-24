"""Organization configuration and reproducible results."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from proteus_music.models.quality import CategoryReport, CountSelection, QualityReport


class OrganizationMode(StrEnum):
    GUIDED = "guided"
    AUTO = "auto"


class Category(BaseModel):
    """A user-named playlist. Required categories are hard constraints; preferred ones are
    created only when the collection supports them."""

    model_config = ConfigDict(frozen=True)

    name: str
    required: bool = True

    @field_validator("name")
    @classmethod
    def _name_required(cls, value: str) -> str:
        value = " ".join(value.split())
        if not value:
            raise ValueError("category names must not be empty")
        return value


class OrganizationConfig(BaseModel):
    """Hard constraints and settings for one organization run."""

    model_config = ConfigDict(frozen=True)

    mode: OrganizationMode = OrganizationMode.GUIDED
    playlists: int | None = Field(default=None, ge=1)
    min_playlists: int | None = Field(default=None, ge=1)
    max_playlists: int | None = Field(default=None, ge=1)
    categories: tuple[Category, ...] = ()
    min_size: int = Field(default=1, ge=1)
    max_size: int | None = Field(default=None, ge=1)
    algorithm: str = "kmeans-merge"
    seed: int = 0

    @property
    def required_categories(self) -> tuple[Category, ...]:
        return tuple(c for c in self.categories if c.required)

    @property
    def preferred_categories(self) -> tuple[Category, ...]:
        return tuple(c for c in self.categories if not c.required)

    @property
    def has_range(self) -> bool:
        return self.min_playlists is not None or self.max_playlists is not None

    def count_bounds(self) -> tuple[int, int | None]:
        """Allowed total playlist count before size constraints (None = no upper bound)."""
        n_required = len(self.required_categories)
        if self.playlists is not None:
            return self.playlists, self.playlists
        if self.has_range:
            return max(self.min_playlists or 1, n_required, 1), self.max_playlists
        if self.mode is OrganizationMode.AUTO:
            return max(n_required, 1), None
        return n_required, n_required + len(self.preferred_categories)

    @model_validator(mode="after")
    def _consistent(self) -> OrganizationConfig:
        errors = []
        if self.max_size is not None and self.max_size < self.min_size:
            errors.append(f"max_size ({self.max_size}) is below min_size ({self.min_size})")
        if self.playlists is not None and self.has_range:
            errors.append("give either an exact playlist count or a range, not both")
        low, high = self.min_playlists, self.max_playlists
        if low is not None and high is not None and high < low:
            errors.append(f"max_playlists ({high}) is below min_playlists ({low})")
        if self.mode is OrganizationMode.AUTO and (
            self.playlists is not None or self.has_range or self.required_categories
        ):
            errors.append(
                "auto mode chooses the count and groups itself; drop the playlist count and "
                "required categories (preferred categories are allowed)"
            )
        if self.mode is OrganizationMode.GUIDED and not (
            self.playlists is not None or self.has_range or self.required_categories
        ):
            errors.append(
                "specify a playlist count, a playlist range or required categories, "
                "or use auto mode"
            )
        names = [" ".join(c.name.casefold().split()) for c in self.categories]
        if len(set(names)) != len(names):
            errors.append("category names must be unique")
        n_required = len(self.required_categories)
        upper = self.playlists if self.playlists is not None else self.max_playlists
        if upper is not None and n_required > upper:
            errors.append(f"{n_required} required categories do not fit in {upper} playlist(s)")
        if errors:
            raise ValueError("; ".join(errors))
        return self


class AssignmentFlag(StrEnum):
    MISSING_FEATURES = "missing-features"
    PARTIAL_FEATURES = "partial-features"
    LOW_SIMILARITY = "low-similarity"
    AMBIGUOUS = "ambiguous"
    CONSTRAINT_DISPLACED = "constraint-displaced"


class PlaylistOrigin(StrEnum):
    DISCOVERED = "discovered"
    REQUIRED_CATEGORY = "required-category"
    PREFERRED_CATEGORY = "preferred-category"
    UNCLASSIFIED = "unclassified"


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
    origin: PlaylistOrigin = PlaylistOrigin.DISCOVERED


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
    quality: QualityReport | None = None
    selection: CountSelection | None = None
    categories: tuple[CategoryReport, ...] = ()

    @property
    def low_confidence(self) -> tuple[Assignment, ...]:
        return tuple(a for a in self.assignments if a.flags)
