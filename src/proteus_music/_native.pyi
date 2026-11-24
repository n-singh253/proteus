from os import PathLike

class AudioFile:
    path: str
    format: str
    size_bytes: int
    modified_unix_ms: int | None
    duration_ms: int | None
    title: str | None
    artists: list[str]
    album: str | None
    album_artist: str | None
    genres: list[str]
    year: int | None
    track_number: int | None
    isrc: str | None
    musicbrainz_recording_id: str | None

class FileFailure:
    path: str
    reason: str

def scan_library(
    root: str | PathLike[str], recursive: bool = True, max_threads: int | None = None
) -> tuple[list[AudioFile], list[FileFailure]]: ...
def read_metadata(
    paths: list[str | PathLike[str]], max_threads: int | None = None
) -> tuple[list[AudioFile], list[FileFailure]]: ...
