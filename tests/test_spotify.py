"""Spotify adapter against a mocked Web API: auth, paging, limits, partial failures, resume."""

import base64
import hashlib
import json
import threading
import urllib.request
from urllib.parse import parse_qs, urlsplit

import pytest

from proteus_music.engine.features.metadata import describe
from proteus_music.integrations.contract import ImportBatch, MatchStatus
from proteus_music.integrations.http import HttpResponse
from proteus_music.integrations.spotify import auth
from proteus_music.integrations.spotify.client import RateLimitedError, SpotifyClient
from proteus_music.integrations.spotify.export import preview, publish
from proteus_music.integrations.spotify.importer import import_library
from proteus_music.library.service import import_batch
from proteus_music.models.organization import OrganizationConfig
from proteus_music.models.track import TrackRecord
from proteus_music.playlists.service import organize_library
from proteus_music.storage import results as result_store
from proteus_music.storage.db import connect
from proteus_music.storage.tracks import load_tracks, remove_source


def track(i, *, isrc=None, title=None, duration=200_000):
    return {
        "type": "track", "id": f"sp{i}", "name": title or f"Song {i}", "duration_ms": duration,
        "artists": [{"name": "Artist"}], "album": {"name": "LP", "release_date": "2001-05-01"},
        "external_ids": {"isrc": isrc} if isrc else {}, "is_local": False,
    }  # fmt: skip


class MemoryStore:
    def __init__(self, scopes=auth.READ_SCOPES, expires_at=1e12):
        self.saved = auth.Credentials("cid", auth.DEFAULT_REDIRECT, "old", "r1", expires_at, scopes)

    def load(self):
        return self.saved

    def save(self, credentials):
        self.saved = credentials

    def delete(self):
        self.saved = None
        return True


class FakeSpotify:
    """In-memory Web API; ``fail`` maps (method, path) to queued responses served first."""

    def __init__(self, saved=(), playlists=(), search=None):
        self.saved, self.search, self.fail = list(saved), search or {}, {}
        self.playlists = {p["id"]: p for p in playlists}
        self.calls = []

    def request(self, method, url, headers, body, timeout):
        parts = urlsplit(url)
        path, query = parts.path.removeprefix("/v1"), parse_qs(parts.query)
        self.calls.append((method, path))
        queued = self.fail.get((method, path))
        if queued:
            response = queued.pop(0)
            if isinstance(response, Exception):
                raise response
            return response
        if path == "/api/token":
            return self._json({"access_token": "new", "refresh_token": "r2", "expires_in": 3600})
        if headers.get("Authorization") == "Bearer old" and path == "/me":
            return HttpResponse(401, b"")
        payload = json.loads(body) if body else None
        if path == "/me":
            return self._json({"id": "me"})
        if path == "/me/tracks":
            return self._page(url, [{"track": t} for t in self.saved], query)
        if path == "/search":
            isrc = query["q"][0].removeprefix("isrc:")
            return self._json({"tracks": {"items": self.search.get(isrc, [])}})
        if path == "/me/playlists" and method == "POST":
            pid = f"pl{len(self.playlists)}"
            self.playlists[pid] = {**payload, "id": pid, "owner": {"id": "me"}, "uris": []}
            return self._json(self.playlists[pid], 201)
        if path == "/me/playlists":
            listed = [{**p, "items": {"total": len(p["uris"])}} for p in self.playlists.values()]
            return self._page(url, listed, query)
        pid = path.split("/")[2]
        if method == "POST":
            self.playlists[pid]["uris"].extend(payload["uris"])
            return self._json({"snapshot_id": "s"}, 201)
        return self._page(url, [{"item": t} for t in self.playlists[pid]["tracks"]], query)

    def _json(self, data, status=200):
        return HttpResponse(status, json.dumps(data).encode())

    def _page(self, url, items, query):
        offset, limit = int(query.get("offset", ["0"])[0]), int(query["limit"][0])
        nxt = None
        if offset + limit < len(items):
            nxt = f"{url.split('?')[0]}?offset={offset + limit}&limit={limit}"
        return self._json({"items": items[offset : offset + limit], "next": nxt})


def quota_exceeded():
    body = {"error": {"status": 429, "message": "Too many requests", "reason": "QUOTA_EXCEEDED"}}
    return HttpResponse(429, json.dumps(body).encode())


def test_pkce_callback_refresh_and_rate_limits():
    request = auth.authorization_request("cid", "http://127.0.0.1:8799/cb", auth.READ_SCOPES)
    query = parse_qs(urlsplit(request.url).query)
    digest = hashlib.sha256(request.verifier.encode()).digest()
    assert query["code_challenge"][0] == base64.urlsafe_b64encode(digest).rstrip(b"=").decode()
    forged = threading.Timer(
        0.3, urllib.request.urlopen, ["http://127.0.0.1:8799/cb?code=c&state=forged"]
    )
    forged.start()
    with pytest.raises(auth.AuthorizationError, match="wrong state"):
        auth.wait_for_code("http://127.0.0.1:8799/cb", request.state, timeout=5)

    fake, store, slept = FakeSpotify(), MemoryStore(), []
    client = SpotifyClient(fake, store, sleep=slept.append)
    fake.fail[("GET", "/me")] = [HttpResponse(429, b"", retry_after=3.0)]
    assert client.me() == {"id": "me"}  # 429 honoured, then 401 -> refresh -> retry.
    assert slept == [3.0] and store.saved.refresh_token == "r2"
    fake.fail[("GET", "/me")] = [quota_exceeded()]
    with pytest.raises(RateLimitedError, match="quota"):
        client.me()
    fake.fail[("POST", "/me/playlists")] = [HttpResponse(502, b"")]
    with pytest.raises(Exception, match="did not confirm"):
        client.create_playlist("x", "")  # An unconfirmed write is never blindly retried.
    assert fake.calls.count(("POST", "/me/playlists")) == 1


def test_import_pages_restricts_analysis_and_reports_quota_stop(tmp_path):
    episode = {**track(999), "type": "episode"}
    local = {**track(998), "id": None, "is_local": True}
    playlists = [
        {"id": "a", "name": "Mine", "owner": {"id": "me"}, "uris": [],
         "tracks": [track(0), track(130), episode, local]},
        {"id": "b", "name": "Theirs", "owner": {"id": "other"}, "uris": [], "tracks": []},
    ]  # fmt: skip
    fake = FakeSpotify(saved=[track(i) for i in range(120)], playlists=playlists)
    batch = import_library(SpotifyClient(fake, MemoryStore()))
    assert len(batch.records) == 121 and len(batch.failures) == 2  # Episode, local file.
    assert fake.calls.count(("GET", "/me/tracks")) == 3  # 50 + 50 + 20.
    assert any("not yours" in w.reason for w in batch.warnings)
    conn = connect(tmp_path / "lib.db")
    import_batch(conn, batch)
    spotify_only = load_tracks(conn)[0]
    assert not spotify_only.records[0].analysis_permitted
    assert describe(spotify_only).artists == () and describe(spotify_only).year is None

    next_page = "https://api.spotify.com/v1/me/tracks?offset=50&limit=50"
    first_page = fake._json({"items": [{"track": track(1)}], "next": next_page})
    fake.fail[("GET", "/me/tracks")] = [first_page, quota_exceeded()]
    partial = import_library(SpotifyClient(fake, MemoryStore()), playlists=False)
    assert len(partial.records) == 1 and "quota" in partial.failures[-1].reason


def _library(tmp_path):
    def local(i, **extra):
        return TrackRecord(source="csv", source_key=str(i), title=f"Song {i}", artists=("Artist",),
                           duration_ms=200_000, genres=("Jazz",), **extra)  # fmt: skip

    conn = connect(tmp_path / "lib.db")
    import_batch(conn, ImportBatch("csv", (
        local(1), local(2, isrc="USAAA0000002"), local(3, isrc="USAAA0000003"), local(4),
    )))  # fmt: skip
    fake = FakeSpotify(
        saved=[track(1)],
        search={"USAAA0000002": [track(2, isrc="USAAA0000002")],
                "USAAA0000003": [track(3, isrc="USAAA0000003", title="Song 3 (Live)")]},
    )  # fmt: skip
    import_batch(conn, import_library(SpotifyClient(fake, MemoryStore()), playlists=False))
    return conn, fake


def test_preview_and_publish_resume_without_duplicate_playlists(tmp_path):
    conn, fake = _library(tmp_path)
    result = organize_library(conn, OrganizationConfig(playlists=1))
    tracks = {t.id: t for t in load_tracks(conn)}
    offline = preview(result, tracks)
    assert offline.count(MatchStatus.MATCHED) == 1 and offline.count(MatchStatus.UNAVAILABLE) == 3
    plan = preview(result, tracks, SpotifyClient(fake, MemoryStore()))
    status = {tracks[m.track_id].title: m.status for m in plan.playlists[0].matches}
    assert status == {"Song 1": "matched", "Song 2": "matched", "Song 3": "ambiguous",
                      "Song 4": "unavailable"}  # fmt: skip
    assert not any(c[0] == "POST" for c in fake.calls)  # Previews never write.

    with pytest.raises(ValueError, match="confirmation"):
        publish(SpotifyClient(fake, MemoryStore()), plan, confirmed=False)
    with pytest.raises(auth.AuthorizationError, match="read-only"):
        publish(SpotifyClient(fake, MemoryStore()), plan, confirmed=True)
    writer = SpotifyClient(fake, MemoryStore(scopes=auth.READ_SCOPES + auth.WRITE_SCOPES))
    fake.fail[("POST", "/playlists/pl0/items")] = [OSError("connection reset")]
    first = publish(writer, plan, confirmed=True)
    assert not first.complete and "incomplete" in first.playlists[0].detail
    second = publish(writer, plan, confirmed=True)  # Retry adopts the marked playlist.
    assert second.complete and second.playlists[0].detail == "resumed"
    assert len(fake.playlists) == 1
    assert fake.playlists["pl0"]["uris"] == list(plan.playlists[0].references)
    assert publish(writer, plan, confirmed=True).playlists[0].detail == "already published"


def test_disconnect_deletes_spotify_data_and_dependent_results(tmp_path):
    conn, fake = _library(tmp_path)
    fake.saved.append(track(9))  # Known only from Spotify.
    import_batch(conn, import_library(SpotifyClient(fake, MemoryStore()), playlists=False))
    organize_library(conn, OrganizationConfig(playlists=1))
    dry = remove_source(conn, "spotify", apply=False)
    assert (dry.records, dry.tracks_removed, dry.tracks_rebuilt, dry.results_removed) == (
        2,
        1,
        1,
        1,
    )
    assert len(load_tracks(conn)) == 5  # A dry run changes nothing.
    remove_source(conn, "spotify", apply=True)
    tracks = load_tracks(conn)
    assert len(tracks) == 4 and all(r.source == "csv" for t in tracks for r in t.records)
    assert tracks[0].album is None and tracks[0].year is None  # Spotify-copied fields gone.
    assert result_store.latest_result_id(conn) is None
