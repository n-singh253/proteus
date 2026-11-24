//! Bounded audio decoding (Symphonia) for acoustic features and fingerprints.

use std::fs::File;
use std::path::{Path, PathBuf};

use pyo3::prelude::*;
use pyo3::types::PyBytes;
use rayon::prelude::*;
use symphonia::core::codecs::audio::AudioDecoderOptions;
use symphonia::core::errors::Error;
use symphonia::core::formats::probe::Hint;
use symphonia::core::formats::{FormatOptions, TrackType};
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;

use crate::metadata::FileFailure;
use crate::scan::with_pool;

/// A mono excerpt decoded at the file's native sample rate.
#[pyclass(frozen, module = "proteus_music._native")]
pub struct DecodedAudio {
    #[pyo3(get)]
    pub path: String,
    #[pyo3(get)]
    pub codec: String,
    #[pyo3(get)]
    pub sample_rate: u32,
    #[pyo3(get)]
    pub channels: u32,
    mono: Vec<f32>,
}

#[pymethods]
impl DecodedAudio {
    /// Little-endian float32 samples, for `numpy.frombuffer(..., dtype="<f4")`.
    #[getter]
    fn samples<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        let bytes: Vec<u8> = self.mono.iter().flat_map(|s| s.to_le_bytes()).collect();
        PyBytes::new(py, &bytes)
    }
}

pub struct Stream {
    pub codec: String,
    pub sample_rate: u32,
    pub channels: u32,
}

/// Decode `path`, calling `sink` with interleaved f32 frames until it returns false.
pub fn decode_with<F>(path: &Path, mut sink: F) -> Result<Stream, String>
where
    F: FnMut(&[f32], u32, u32) -> bool,
{
    let file = File::open(path).map_err(|e| e.to_string())?;
    let mss = MediaSourceStream::new(Box::new(file), Default::default());
    let mut hint = Hint::new();
    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
        hint.with_extension(ext);
    }
    let mut format = symphonia::default::get_probe()
        .probe(
            &hint,
            mss,
            FormatOptions::default(),
            MetadataOptions::default(),
        )
        .map_err(|e| format!("unsupported container: {e}"))?;
    let track = format
        .default_track(TrackType::Audio)
        .ok_or("no audio track")?;
    let params = track
        .codec_params
        .as_ref()
        .and_then(|p| p.audio())
        .ok_or("no audio codec parameters")?
        .clone();
    let registry = symphonia::default::get_codecs();
    let codec = registry
        .get_audio_decoder(params.codec)
        .map(|r| r.codec.info.short_name.to_owned())
        .ok_or_else(|| format!("no decoder for codec {:?}", params.codec))?;
    let mut decoder = registry
        .make_audio_decoder(&params, &AudioDecoderOptions::default())
        .map_err(|e| format!("cannot decode {codec}: {e}"))?;
    let track_id = track.id;
    let mut stream = Stream {
        codec,
        sample_rate: 0,
        channels: 0,
    };
    let mut interleaved: Vec<f32> = Vec::new();
    loop {
        let packet = match format.next_packet() {
            Ok(Some(packet)) => packet,
            Ok(None) => break,
            Err(Error::IoError(e)) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(e) => return Err(format!("demux error: {e}")),
        };
        if packet.track_id != track_id {
            continue;
        }
        let buffer = match decoder.decode(&packet) {
            Ok(buffer) => buffer,
            Err(Error::DecodeError(_)) => continue, // Skip a corrupt packet, as players do.
            Err(e) => return Err(format!("decode error: {e}")),
        };
        stream.sample_rate = buffer.spec().rate();
        stream.channels = buffer.spec().channels().count() as u32;
        interleaved.clear();
        buffer.copy_to_vec_interleaved(&mut interleaved);
        if !sink(&interleaved, stream.sample_rate, stream.channels) {
            break;
        }
    }
    if stream.sample_rate == 0 {
        return Err("no decodable audio".to_owned());
    }
    Ok(stream)
}

fn excerpt(path: &Path, offset_s: f64, max_s: f64) -> Result<DecodedAudio, String> {
    let mut mono = Vec::new();
    let mut skipped = 0usize;
    let stream = decode_with(path, |frames, rate, channels| {
        let ch = channels.max(1) as usize;
        let (skip, keep) = (
            (offset_s * rate as f64) as usize,
            (max_s * rate as f64) as usize,
        );
        for frame in frames.chunks_exact(ch) {
            if skipped < skip {
                skipped += 1;
                continue;
            }
            mono.push(frame.iter().sum::<f32>() / ch as f32);
            if mono.len() >= keep {
                return false;
            }
        }
        true
    })?;
    if mono.is_empty() {
        return Err("no audio after the analysis offset".to_owned());
    }
    Ok(DecodedAudio {
        path: path.to_string_lossy().into_owned(),
        codec: stream.codec,
        sample_rate: stream.sample_rate,
        channels: stream.channels,
        mono,
    })
}

/// Decode a batch of mono excerpts in parallel: `max_seconds` from each file's offset.
#[pyfunction]
#[pyo3(signature = (paths, offsets, max_seconds, max_threads = None))]
pub fn decode_audio(
    py: Python<'_>,
    paths: Vec<PathBuf>,
    offsets: Vec<f64>,
    max_seconds: f64,
    max_threads: Option<usize>,
) -> PyResult<(Vec<DecodedAudio>, Vec<FileFailure>)> {
    if paths.len() != offsets.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "paths and offsets differ in length",
        ));
    }
    py.detach(|| {
        with_pool(max_threads, || {
            let results: Vec<_> = paths
                .par_iter()
                .zip(offsets.par_iter())
                .map(|(p, &o)| excerpt(p, o, max_seconds).map_err(|r| (p.clone(), r)))
                .collect();
            split(results)
        })
    })
    .map_err(pyo3::exceptions::PyRuntimeError::new_err::<String>)
}

pub fn split<T>(results: Vec<Result<T, (PathBuf, String)>>) -> (Vec<T>, Vec<FileFailure>) {
    let mut ok = Vec::new();
    let mut failed = Vec::new();
    for result in results {
        match result {
            Ok(value) => ok.push(value),
            Err((path, reason)) => failed.push(FileFailure {
                path: path.to_string_lossy().into_owned(),
                reason,
            }),
        }
    }
    (ok, failed)
}
