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


def test_reliable_identifiers_and_exact_metadata_merge():
    index = IdentityIndex()
    index.add("t1", record("a", "Song", duration_ms=200_000, isrc="US-AAA-00-00001"))

    assert index.resolve(record("b", "Totally Different", isrc="USAAA0000001")) == "t1"
    assert index.resolve(record("c", "SONG (feat. Guest)", duration_ms=201_500)) == "t1"


def test_parse_title_extracts_featured_artists_and_version():
    parts = parse_title("Track (feat. A & B) [Radio Edit]")
    assert (parts.base, parts.version, parts.featured) == ("Track", "radio edit", ("A", "B"))
