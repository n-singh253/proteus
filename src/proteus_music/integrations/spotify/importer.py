"""Read-only import of a user's Spotify saved tracks and own playlists.

Spotify's Developer Policy forbids analyzing Spotify Content or using it to train models, so
every record is identification-only (``analysis_permitted=False``): it resolves identity and
enables export, but never reaches feature extraction. Records carry no genres or tags.
"""

from __future__ import annotations

from typing import Any

from proteus_music.integrations.contract import AdapterInfo, Capability, ImportBatch, ImportIssue
from proteus_music.integrations.spotify.client import SpotifyClient, SpotifyError
from proteus_music.models.track import TrackRecord

SOURCE = "spotify"

ADAPTER = AdapterInfo(
    name=SOURCE,
    capabilities=frozenset({Capability.IMPORT, Capability.MATCH, Capability.EXPORT}),
    description=(
        "Your own Spotify development-mode app: read saved tracks and own playlists "
        "(identification only), preview matches, and create private playlists after confirmation."
    ),
)


def track_uri(source_key: str) -> str:
    return f"spotify:track:{source_key}"


def to_record(track: dict[str, Any]) -> TrackRecord | ImportIssue:
    """Normalize one Spotify track object, or explain why it cannot be imported."""
    name = track.get("name") or "(untitled)"
    if track.get("type", "track") != "track":
        return ImportIssue(name, f"skipped {track.get('type')}: only music tracks are imported")
    if track.get("is_local") or not track.get("id"):
        return ImportIssue(
            name, "Spotify local file without a catalog ID; import the file itself instead"
        )
    album = track.get("album") or {}
    released = str(album.get("release_date") or "")[:4]
    try:
        return TrackRecord(
            source=SOURCE,
            source_key=track["id"],
            title=name,
            artists=tuple(a["name"] for a in track.get("artists") or () if a.get("name")),
            album=album.get("name"),
            duration_ms=track.get("duration_ms") or None,
            year=int(released) if released.isdigit() and released != "0000" else None,
            isrc=(track.get("external_ids") or {}).get("isrc"),
            analysis_permitted=False,
        )
    except ValueError as error:
        return ImportIssue(name, f"invalid track data: {error}")


def import_library(
    client: SpotifyClient, *, saved: bool = True, playlists: bool = True
) -> ImportBatch:
    """Records from saved tracks and owned or collaborative playlists, deduplicated by ID.

    Throttling or a spent quota stops reading; what was read is still returned (imports are
    idempotent, so re-running later completes it) and the stop is reported as a failure.
    """
    records: dict[str, TrackRecord] = {}
    failures: list[ImportIssue] = []
    warnings: list[ImportIssue] = []

    def take(location: str, track: dict[str, Any] | None) -> None:
        if not track:
            failures.append(ImportIssue(location, "unavailable item (removed from Spotify)"))
            return
        outcome = to_record(track)
        if isinstance(outcome, ImportIssue):
            failures.append(ImportIssue(f"{location}: {outcome.location}", outcome.reason))
            return
        if track.get("is_playable") is False or track.get("restrictions"):
            warnings.append(ImportIssue(outcome.title, "not playable in your market"))
        records.setdefault(outcome.source_key, outcome)

    try:
        if saved:
            for entry in client.paginate("/me/tracks"):
                take("saved tracks", entry.get("track"))
        if playlists:
            me = client.me().get("id")
            for playlist in client.paginate("/me/playlists"):
                label = f"playlist '{playlist.get('name')}'"
                owner = (playlist.get("owner") or {}).get("id")
                if owner != me and not playlist.get("collaborative"):
                    warnings.append(
                        ImportIssue(label, "not yours; Spotify only lists its items to owners")
                    )
                    continue
                for entry in client.paginate(f"/playlists/{playlist['id']}/items"):
                    take(label, entry.get("item") or entry.get("track"))
    except SpotifyError as error:
        failures.append(ImportIssue("spotify", f"stopped early: {error}"))
    return ImportBatch(SOURCE, tuple(records.values()), tuple(failures), tuple(warnings))
