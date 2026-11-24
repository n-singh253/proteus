"""Typed request and response models for the local API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from proteus_music.api.jobs import Job, JobState
from proteus_music.models.organization import OrganizationResult, PlaylistOrigin
from proteus_music.models.quality import CategoryReport, CountSelection, QualityReport
from proteus_music.models.track import Track

TRACKS_PER_PLAYLIST = 200


class ImportRequest(BaseModel):
    kind: Literal["csv", "json", "m3u", "local"]
    path: str
    identification_only: bool = False
    recursive: bool = True
    separator: str = Field(default=";", min_length=1, max_length=3)


class ExportRequest(BaseModel):
    target: Literal["m3u8", "csv", "json"]
    out_dir: str
    relative_paths: bool = False
    overwrite: bool = False


class JobView(BaseModel):
    id: str
    kind: str
    state: JobState
    stage: str
    progress: float | None
    result: dict | None
    error: str | None
    reasons: list[str]

    @classmethod
    def of(cls, job: Job) -> JobView:
        return cls(**{k: getattr(job, k) for k in cls.model_fields})


class TrackView(BaseModel):
    id: str
    title: str
    artists: list[str]
    flags: list[str]
    similarity: float


class PlaylistView(BaseModel):
    index: int
    name: str
    description: str
    origin: PlaylistOrigin
    size: int
    cohesion: float
    flagged: int
    tracks: list[TrackView]  # At most TRACKS_PER_PLAYLIST, best-fitting first.
    flagged_tracks: list[TrackView]  # Flagged members, at most TRACKS_PER_PLAYLIST.


class ResultView(BaseModel):
    id: str
    created_at: str
    track_count: int
    config: dict
    warnings: list[str]
    quality: QualityReport | None
    selection: CountSelection | None
    categories: list[CategoryReport]
    playlists: list[PlaylistView]

    @classmethod
    def of(cls, result: OrganizationResult, tracks: dict[str, Track]) -> ResultView:
        by_track = {a.track_id: a for a in result.assignments}

        def track(track_id: str) -> TrackView:
            t, a = tracks[track_id], by_track[track_id]
            return TrackView(id=t.id, title=t.title, artists=list(t.artists),
                             flags=list(a.flags), similarity=a.similarity)  # fmt: skip

        playlists = [
            PlaylistView(
                index=p.index,
                name=p.name,
                description=p.description,
                origin=p.origin,
                size=len(p.track_ids),
                cohesion=p.cohesion,
                flagged=sum(1 for t in p.track_ids if by_track[t].flags),
                tracks=[track(t) for t in p.track_ids[:TRACKS_PER_PLAYLIST]],
                flagged_tracks=[track(t) for t in p.track_ids if by_track[t].flags][
                    :TRACKS_PER_PLAYLIST
                ],
            )
            for p in result.playlists
        ]
        return cls(
            id=result.id,
            created_at=result.created_at.isoformat(),
            track_count=len(result.snapshot_track_ids),
            config=result.config.model_dump(mode="json"),
            warnings=list(result.warnings),
            quality=result.quality,
            selection=result.selection,
            categories=list(result.categories),
            playlists=playlists,
        )
