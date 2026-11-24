//! Chromaprint-compatible audio fingerprints (rusty-chromaprint, default configuration).

use std::path::{Path, PathBuf};

use pyo3::prelude::*;
use rayon::prelude::*;
use rusty_chromaprint::{Configuration, Fingerprinter};

use crate::decode::{decode_with, split};
use crate::metadata::FileFailure;
use crate::scan::with_pool;

type Fingerprints = (Vec<(String, Vec<u32>)>, Vec<FileFailure>);

fn fingerprint(path: &Path, max_s: f64) -> Result<(String, Vec<u32>), String> {
    let config = Configuration::preset_test2();
    let mut printer = Fingerprinter::new(&config);
    let mut started = false;
    let mut frames_seen = 0usize;
    let mut pcm: Vec<i16> = Vec::new();
    decode_with(path, |frames, rate, channels| {
        if !started {
            started = printer.start(rate, channels).is_ok();
        }
        pcm.clear();
        pcm.extend(
            frames
                .iter()
                .map(|s| (s.clamp(-1.0, 1.0) * i16::MAX as f32) as i16),
        );
        printer.consume(&pcm);
        frames_seen += frames.len() / channels.max(1) as usize;
        started && (frames_seen as f64) < max_s * rate as f64
    })?;
    if !started {
        return Err("fingerprinter rejected the stream format".to_owned());
    }
    printer.finish();
    let print = printer.fingerprint().to_vec();
    if print.is_empty() {
        return Err("audio too short to fingerprint".to_owned());
    }
    Ok((path.to_string_lossy().into_owned(), print))
}

/// Fingerprint the first `max_seconds` of each file, in parallel.
#[pyfunction]
#[pyo3(signature = (paths, max_seconds = 120.0, max_threads = None))]
pub fn fingerprint_audio(
    py: Python<'_>,
    paths: Vec<PathBuf>,
    max_seconds: f64,
    max_threads: Option<usize>,
) -> PyResult<Fingerprints> {
    py.detach(|| {
        with_pool(max_threads, || {
            let results: Vec<_> = paths
                .par_iter()
                .map(|p| fingerprint(p, max_seconds).map_err(|r| (p.clone(), r)))
                .collect();
            split(results)
        })
    })
    .map_err(pyo3::exceptions::PyRuntimeError::new_err::<String>)
}
