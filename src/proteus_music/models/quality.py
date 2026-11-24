"""Quality, playlist-count selection and category-evidence reports attached to results."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict


class QualityReport(BaseModel):
    """Clustering quality and practical usefulness of one organization, kept separate.

    Clustering metrics use analyzable tracks only (unclassified excluded); usefulness metrics
    cover every track. ``score`` is the selection objective described in the README.
    """

    model_config = ConfigDict(frozen=True)

    # Clustering quality.
    silhouette: float | None
    cohesion: float
    separation: float | None
    compatibility: float
    # Practical usefulness.
    smallest_playlist: int
    largest_playlist: int
    fragmentation: float
    low_confidence_rate: float
    semantic_coherence: float
    analyzable_share: float  # Share of tracks with core (genre or tag) metadata.
    score: float


class CountCandidate(BaseModel):
    model_config = ConfigDict(frozen=True)

    playlists: int
    score: float
    silhouette: float | None
    fragmentation: float
    low_confidence_rate: float


class CountSelection(BaseModel):
    """How the playlist count was chosen within the allowed range."""

    model_config = ConfigDict(frozen=True)

    low: int
    high: int
    chosen: int
    evaluated: tuple[CountCandidate, ...]
    sample_size: int
    reason: str


class Evidence(StrEnum):
    STRONG = "strong"
    WEAK = "weak"
    NONE = "none"


class CategoryReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    required: bool
    matched_terms: tuple[str, ...]
    supporting_tracks: int
    evidence: Evidence
    kept: bool
    note: str = ""
