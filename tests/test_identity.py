import pytest

from proteus_music.library.identity import IdentityIndex, parse_title
from proteus_music.models.track import TrackRecord


def record(key: str, title: str, **fields) -> TrackRecord:
    return TrackRecord(source="test", source_key=key, title=title, artists=("Artist",), **fields)


def test_versions_and_conflicting_signals_stay_separate():
    index = IdentityIndex()
    index.add("studio", record("a", "Song", duration_ms=200_000, isrc="USAAA0000001"))

    assert index.resolve(record("b", "Song (Live at Wembley)", duration_ms=200_000)) is None
    assert index.resolve(record("c", "Song - 2011 Remaster", duration_ms=200_000)) is None
    assert index.resolve(record("d", "Song", duration_ms=260_000)) is None
    assert index.resolve(record("e", "Song", duration_ms=200_000, isrc="USBBB0000002")) is None
    assert index.resolve(record("f", "Song", duration_ms=None)) is None
    assert index.resolve(record("g", "Song (Live)", isrc="USAAA0000001")) is None
    assert "kept separate" in index.conflicts[0]


def test_reliable_identifiers_and_exact_metadata_merge():
    index = IdentityIndex()
    index.add("t1", record("a", "Song", duration_ms=200_000, isrc="US-AAA-00-00001"))

    assert index.resolve(record("b", "Totally Different", isrc="USAAA0000001")) == "t1"
    assert index.resolve(record("c", "SONG (feat. Guest)", duration_ms=201_500)) == "t1"


def test_parse_title_extracts_featured_artists_and_version():
    parts = parse_title("Track (feat. A & B) [Radio Edit]")
    assert (parts.base, parts.version, parts.featured) == ("Track", "radio edit", ("A", "B"))


@pytest.mark.parametrize("order", [("studio", "live"), ("live", "studio")])
def test_shared_isrc_attaches_to_the_matching_version(order):
    titles = {"studio": "Song", "live": "Song (Live)"}
    index, ids = IdentityIndex(), {}
    for name in order:
        candidate = record(name, titles[name], isrc="USAAA0000001")
        ids[name] = index.resolve(candidate) or name
        index.add(ids[name], candidate)

    assert ids == {"studio": "studio", "live": "live"}  # Shared ISRC, distinct recordings.
    assert index.resolve(record("x", "Song (Live)", isrc="USAAA0000001")) == "live"
    assert index.resolve(record("y", "Song", isrc="USAAA0000001")) == "studio"


def test_ambiguous_identifiers_stay_separate():
    index = IdentityIndex()
    index.add("by-isrc", record("a", "Song", isrc="USAAA0000001"))
    index.add("by-mbid", record("b", "Song", musicbrainz_recording_id="mbid-1"))
    assert (
        index.resolve(record("c", "Song", isrc="USAAA0000001", musicbrainz_recording_id="mbid-1"))
        is None
    )
    assert "different tracks" in index.conflicts[-1]

    index.add("duplicate", record("d", "Song", isrc="USAAA0000001"))  # Same ISRC and version.
    assert index.resolve(record("e", "Song", isrc="USAAA0000001")) is None
    assert "shared by 2 track(s)" in index.conflicts[-1]
