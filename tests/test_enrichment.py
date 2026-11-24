"""MusicBrainz enrichment against a mocked web service (no network)."""

import json
from datetime import UTC, datetime, timedelta

from typer.testing import CliRunner

from proteus_music.cli.main import app
from proteus_music.engine.features.metadata import describe
from proteus_music.integrations.contract import ImportBatch
from proteus_music.integrations.musicbrainz.client import HttpResponse, MusicBrainzClient
from proteus_music.library.enrichment import enrich, plan_enrichment
from proteus_music.library.service import import_batch
from proteus_music.models.track import TrackRecord
from proteus_music.storage.db import connect
from proteus_music.storage.enrichment import status_counts
from proteus_music.storage.tracks import load_tracks

NOW = datetime(2026, 11, 1, tzinfo=UTC)


def recording(mbid, title, artist="Alpha", length=200_000, disambiguation="", year="1999"):
    return {"id": mbid, "title": title, "length": length, "disambiguation": disambiguation,
            "artist-credit": [{"name": artist}], "releases": [{"date": f"{year}-05-01"}],
            "isrcs": [], "tags": [{"name": "must-not-be-used"}]}  # fmt: skip


RESPONSES = {
    "EXACT": [recording("mb-exact", "Exact Song")],
    "AMBIG": [recording("mb-a1", "Twin Song"), recording("mb-a2", "Twin Song")],
    "VERSN": [recording("mb-live", "Studio Song", disambiguation="live, 1999-05-01")],
}


class FakeTransport:
    def __init__(self, throttled=("LIMIT",)):
        self.urls, self.throttled = [], set(throttled)

    def get(self, url, headers, timeout):
        assert headers["User-Agent"].startswith("proteus-music/")
        self.urls.append(url)
        isrc = url.split("isrc/")[1].split("?")[0]
        if isrc in self.throttled:
            return HttpResponse(503, b"")
        if isrc not in RESPONSES and isrc != "LIMIT":
            return HttpResponse(404, b"")
        return HttpResponse(200, json.dumps({"recordings": RESPONSES.get(isrc, [])}).encode())


def client(transport):
    return MusicBrainzClient(transport, "https://example.org/proteus-tests", max_requests=50,
                             sleep=lambda _: None, clock=lambda: 0.0)  # fmt: skip


def seed(db):
    def track(key, title, isrc, permitted=True):
        return TrackRecord(
            source="csv",
            source_key=key,
            title=title,
            artists=("Alpha",),
            duration_ms=200_500,
            isrc=isrc,
            analysis_permitted=permitted,
        )

    records = (
        track("1", "Exact Song", "EXACT", permitted=False),
        track("2", "Twin Song", "AMBIG"),
        track("3", "Studio Song", "VERSN"),
        track("4", "Unknown Song", "MISSN"),
        track("5", "Throttled Song", "LIMIT"),
    )
    conn = connect(db)
    import_batch(conn, ImportBatch("csv", records))
    return conn


def test_outcomes_permissions_and_resume(tmp_path):
    conn = seed(tmp_path / "lib.db")
    first = FakeTransport()
    summary = enrich(conn, client(first), plan_enrichment(conn, NOW, refresh=False), NOW)

    assert dict(summary.outcomes) == {"matched": 1, "ambiguous": 1, "no-match": 1, "not-found": 1}
    assert summary.remaining == 1 and "throttled" in summary.stopped
    assert len(first.urls) == 4 + 4  # Four lookups, then the throttled one: 1 try + 3 retries.

    tracks = {t.title: t for t in load_tracks(conn)}
    exact = tracks["Exact Song"]
    restricted, enriched = sorted(exact.records, key=lambda r: r.source)
    assert not restricted.analysis_permitted  # The original record is never upgraded.
    assert enriched.source == "musicbrainz" and enriched.analysis_permitted
    assert describe(exact).year == 1999 and describe(exact).genres == ()  # No tags used.
    for title in ("Twin Song", "Studio Song", "Unknown Song"):
        assert {r.source for r in tracks[title].records} == {"csv"}  # Kept separate.

    resumed = FakeTransport(throttled=())
    plan = plan_enrichment(conn, NOW + timedelta(hours=1), refresh=False)
    assert [q for _, q in plan.due] == ["isrc:LIMIT"] and plan.fresh == 4
    enrich(conn, client(resumed), plan, NOW + timedelta(hours=1))
    assert len(resumed.urls) == 1  # Completed lookups are not repeated.
    assert status_counts(conn, "musicbrainz")["not-found"] == 2  # LIMIT has no recordings.

    later = plan_enrichment(conn, NOW + timedelta(days=45), refresh=False)
    assert sorted(q for _, q in later.due) == [
        "isrc:AMBIG",
        "isrc:LIMIT",
        "isrc:MISSN",
        "isrc:VERSN",
    ]


def test_network_is_opt_in(tmp_path):
    seed(tmp_path / "lib.db")
    result = CliRunner().invoke(app, ["--db", str(tmp_path / "lib.db"), "enrich", "musicbrainz"])
    assert result.exit_code == 0 and "Dry run: nothing was sent" in result.output
    assert "5 track(s) due" in result.output
