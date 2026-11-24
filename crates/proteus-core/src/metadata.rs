use std::path::Path;
use std::time::UNIX_EPOCH;

use lofty::file::{AudioFile as _, FileType, TaggedFileExt};
use lofty::probe::Probe;
use lofty::tag::{Accessor, ItemKey, Tag};
use pyo3::prelude::*;

/// Container and codec metadata read from an audio file's content, never from its extension.
#[pyclass(frozen, get_all, module = "proteus_music._native")]
#[derive(Clone, Debug)]
pub struct AudioFile {
    pub path: String,
    pub format: String,
    pub size_bytes: u64,
    pub modified_unix_ms: Option<i64>,
    pub duration_ms: Option<u64>,
    pub title: Option<String>,
    pub artists: Vec<String>,
    pub album: Option<String>,
    pub album_artist: Option<String>,
    pub genres: Vec<String>,
    pub year: Option<u16>,
    pub track_number: Option<u32>,
    pub isrc: Option<String>,
    pub musicbrainz_recording_id: Option<String>,
}

#[pyclass(frozen, get_all, module = "proteus_music._native")]
#[derive(Clone, Debug)]
pub struct FileFailure {
    pub path: String,
    pub reason: String,
}

fn supported_format(file_type: FileType) -> Option<&'static str> {
    match file_type {
        FileType::Mpeg => Some("mp3"),
        FileType::Flac => Some("flac"),
        FileType::Wav => Some("wav"),
        FileType::Aac => Some("aac"),
        FileType::Mp4 => Some("m4a"),
        FileType::Vorbis => Some("ogg-vorbis"),
        FileType::Opus => Some("opus"),
        _ => None,
    }
}

fn non_empty(value: Option<&str>) -> Option<String> {
    value
        .map(str::trim)
        .filter(|v| !v.is_empty())
        .map(str::to_owned)
}

fn all_strings(tag: &Tag, key: ItemKey) -> Vec<String> {
    tag.get_strings(key)
        .map(str::trim)
        .filter(|v| !v.is_empty())
        .map(str::to_owned)
        .collect()
}

pub fn read_audio_file(path: &Path) -> Result<AudioFile, String> {
    let fs_meta = std::fs::metadata(path).map_err(|e| e.to_string())?;
    let probe = Probe::open(path)
        .map_err(|e| e.to_string())?
        .guess_file_type()
        .map_err(|e| e.to_string())?;
    let format = probe
        .file_type()
        .and_then(supported_format)
        .ok_or_else(|| "unsupported or unrecognized audio container".to_owned())?;
    let tagged = probe.read().map_err(|e| e.to_string())?;
    let duration = tagged.properties().duration();
    let modified_unix_ms = fs_meta
        .modified()
        .ok()
        .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
        .and_then(|d| i64::try_from(d.as_millis()).ok());

    let mut file = AudioFile {
        path: path.to_string_lossy().into_owned(),
        format: format.to_owned(),
        size_bytes: fs_meta.len(),
        modified_unix_ms,
        duration_ms: (!duration.is_zero()).then_some(duration.as_millis() as u64),
        title: None,
        artists: Vec::new(),
        album: None,
        album_artist: None,
        genres: Vec::new(),
        year: None,
        track_number: None,
        isrc: None,
        musicbrainz_recording_id: None,
    };
    if let Some(tag) = tagged.primary_tag().or_else(|| tagged.first_tag()) {
        file.title = non_empty(tag.title().as_deref());
        file.artists = all_strings(tag, ItemKey::TrackArtist);
        file.album = non_empty(tag.album().as_deref());
        file.album_artist = non_empty(tag.get_string(ItemKey::AlbumArtist));
        file.genres = all_strings(tag, ItemKey::Genre);
        file.year = tag.date().map(|d| d.year);
        file.track_number = tag.track();
        file.isrc = non_empty(tag.get_string(ItemKey::Isrc));
        file.musicbrainz_recording_id = non_empty(tag.get_string(ItemKey::MusicBrainzRecordingId));
    }
    Ok(file)
}
