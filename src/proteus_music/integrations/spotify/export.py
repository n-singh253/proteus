"""Spotify export: a preview of matched, ambiguous and unavailable tracks, then an explicitly
confirmed publish that creates private playlists.

Publishing is safe to retry. Each playlist's description carries a marker naming the result
and playlist; a re-run adopts the marked playlist instead of creating another and resumes
after the items it already holds (items are only ever appended, in plan order).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from proteus_music.integrations.contract import (
    ExportPreview,
    MatchStatus,
    PlaylistPreview,
    TrackMatch,
)
from proteus_music.integrations.spotify import auth
from proteus_music.integrations.spotify.client import (
    ADD_BATCH,
    RateLimitedError,
    SpotifyClient,
    SpotifyError,
)
from proteus_music.integrations.spotify.importer import SOURCE, to_record, track_uri
from proteus_music.library.identity import DURATION_TOLERANCE_MS, identity_key
from proteus_music.models.organization import OrganizationResult
from proteus_music.models.track import Track, TrackRecord

MAX_DESCRIPTION = 300


def marker(result_id: str, index: int) -> str:
    return f"[proteus {result_id}/{index}]"


class _IsrcMatcher:
    """Looks up tracks missing from the imported library by ISRC; results are not stored."""

    def __init__(self, client: SpotifyClient | None) -> None:
        self.client, self.stopped = client, None
        self._cache: dict[str, list[TrackRecord]] = {}

    def match(self, track: Track) -> TrackMatch:
        def result(status: MatchStatus, reason: str, uri: str | None = None) -> TrackMatch:
            return TrackMatch(track.id, status, uri, reason)

        if self.client is None:
            return result(MatchStatus.UNAVAILABLE, "not in your imported Spotify library")
        if not track.isrc:
            return result(MatchStatus.UNAVAILABLE, "no ISRC to search by; titles are not matched")
        if self.stopped:
            return result(MatchStatus.UNAVAILABLE, f"not searched: {self.stopped}")
        if track.isrc not in self._cache:
            try:
                found = self.client.search_isrc(track.isrc)
            except SpotifyError as error:
                self.stopped = str(error)
                return result(MatchStatus.UNAVAILABLE, f"not searched: {error}")
            self._cache[track.isrc] = [
                r for r in map(to_record, found) if isinstance(r, TrackRecord)
            ]
        candidates = self._cache[track.isrc]
        if not candidates:
            return result(MatchStatus.UNAVAILABLE, f"ISRC {track.isrc} not found on Spotify")
        same = [c for c in candidates if _same_recording(track, c)]
        if not same or len({identity_key(c)[:2] for c in same}) > 1:
            found = "; ".join(f"'{c.title}' by {', '.join(c.artists)}" for c in candidates[:3])
            return result(MatchStatus.AMBIGUOUS, f"ISRC {track.isrc} on Spotify differs: {found}")
        return result(MatchStatus.MATCHED, "found by ISRC", track_uri(same[0].source_key))


def _same_recording(track: Track, candidate: TrackRecord) -> bool:
    if identity_key(candidate)[2] != (track.version or ""):
        return False
    if track.duration_ms and candidate.duration_ms:
        return abs(track.duration_ms - candidate.duration_ms) <= DURATION_TOLERANCE_MS
    return True


def preview(
    result: OrganizationResult, tracks: dict[str, Track], client: SpotifyClient | None = None
) -> ExportPreview:
    """Match every assigned track: imported Spotify references first, then (with a client)
    an ISRC search that accepts only the same version at a compatible duration."""
    matcher = _IsrcMatcher(client)

    def match(track: Track) -> TrackMatch:
        refs = [r.source_key for r in track.records if r.source == SOURCE]
        if refs:
            return TrackMatch(track.id, MatchStatus.MATCHED, track_uri(refs[0]), "in your library")
        return matcher.match(track)

    playlists = tuple(
        PlaylistPreview(
            p.index, p.name, p.description, tuple(match(tracks[t]) for t in p.track_ids)
        )
        for p in result.playlists
    )
    return ExportPreview(SOURCE, result.id, playlists)


@dataclass(frozen=True)
class PublishedPlaylist:
    index: int
    name: str
    playlist_id: str | None
    added: int  # Items now in the Spotify playlist that came from this plan.
    planned: int
    detail: str

    @property
    def complete(self) -> bool:
        return self.added == self.planned


@dataclass(frozen=True)
class PublishReport:
    playlists: tuple[PublishedPlaylist, ...]
    stopped: str | None  # Set when throttling or the quota ended the run early.

    @property
    def complete(self) -> bool:
        return self.stopped is None and all(p.complete for p in self.playlists)


def _marked_playlists(client: SpotifyClient, result_id: str) -> dict[str, dict[str, Any]]:
    me = client.me().get("id")
    prefix = marker(result_id, 0).rsplit("/", 1)[0]
    found = {}
    for playlist in client.paginate("/me/playlists"):
        description = playlist.get("description") or ""
        if (playlist.get("owner") or {}).get("id") == me and prefix in description:
            start = description.index(prefix)
            found[description[start : description.index("]", start) + 1]] = playlist
    return found


def _item_total(playlist: dict[str, Any]) -> int:
    return int((playlist.get("items") or playlist.get("tracks") or {}).get("total") or 0)


def _publish_one(
    client: SpotifyClient, playlist: PlaylistPreview, tag: str, remote: dict[str, Any] | None
) -> tuple[PublishedPlaylist, str | None]:
    """Create or adopt one playlist and append the items it does not hold yet.

    Returns the outcome and, when throttling or the quota stopped it, why the run must stop.
    """
    uris, added, detail, stopped = list(playlist.references), 0, "created", None
    try:
        if remote is not None:
            added = _item_total(remote)
            if added > len(uris):
                detail = "changed on Spotify since it was created; left as is"
                return PublishedPlaylist(
                    playlist.index, playlist.name, remote["id"], added, len(uris), detail
                ), None
            detail = "already published" if added == len(uris) else "resumed"
        else:
            text = playlist.description[: MAX_DESCRIPTION - len(tag) - 1]
            remote = client.create_playlist(playlist.name, f"{text} {tag}".strip())
        for start in range(added, len(uris), ADD_BATCH):
            client.add_items(remote["id"], uris[start : start + ADD_BATCH])
            added = min(len(uris), start + ADD_BATCH)
    except SpotifyError as error:
        detail = f"incomplete: {error}; re-run to resume"
        stopped = str(error) if isinstance(error, RateLimitedError) else None
    pid = remote["id"] if remote else None
    return PublishedPlaylist(playlist.index, playlist.name, pid, added, len(uris), detail), stopped


def publish(client: SpotifyClient, plan: ExportPreview, *, confirmed: bool) -> PublishReport:
    """Create (or resume) one private Spotify playlist per previewed playlist with matches."""
    if not confirmed:
        raise ValueError("publishing writes to your Spotify account and needs confirmation")
    if not client.has_scopes(auth.WRITE_SCOPES):
        raise auth.AuthorizationError(
            "this connection is read-only; run `proteus spotify login --write` to allow "
            "creating private playlists"
        )
    existing = _marked_playlists(client, plan.result_id)
    done: list[PublishedPlaylist] = []
    stopped = None
    for playlist in plan.playlists:
        tag, planned = marker(plan.result_id, playlist.index), len(playlist.references)
        if stopped or not planned:
            reason = f"not attempted: {stopped}" if stopped else "no matched tracks"
            done.append(PublishedPlaylist(playlist.index, playlist.name, None, 0, planned, reason))
            continue
        outcome, stopped = _publish_one(client, playlist, tag, existing.get(tag))
        done.append(outcome)
    return PublishReport(tuple(done), stopped)
