use std::path::{Path, PathBuf};

use pyo3::exceptions::{PyNotADirectoryError, PyRuntimeError};
use pyo3::prelude::*;
use rayon::prelude::*;
use walkdir::WalkDir;

use crate::metadata::{AudioFile, FileFailure, read_audio_file};

/// Extensions worth probing. The extension only nominates a candidate; the content decides the format.
const CANDIDATE_EXTENSIONS: &[&str] = &[
    "mp3", "flac", "wav", "aac", "m4a", "mp4", "ogg", "oga", "opus",
];

type ReadOutcome = (Vec<AudioFile>, Vec<FileFailure>);

fn is_candidate(path: &Path) -> bool {
    path.extension().and_then(|e| e.to_str()).is_some_and(|e| {
        CANDIDATE_EXTENSIONS
            .iter()
            .any(|c| c.eq_ignore_ascii_case(e))
    })
}

fn failure(path: &Path, reason: impl Into<String>) -> FileFailure {
    FileFailure {
        path: path.to_string_lossy().into_owned(),
        reason: reason.into(),
    }
}

fn collect_candidates(root: &Path, recursive: bool) -> (Vec<PathBuf>, Vec<FileFailure>) {
    let mut paths = Vec::new();
    let mut failures = Vec::new();
    let walker = WalkDir::new(root).max_depth(if recursive { usize::MAX } else { 1 });
    for entry in walker {
        match entry {
            Ok(e) if e.file_type().is_file() && is_candidate(e.path()) => paths.push(e.into_path()),
            Ok(_) => {}
            Err(e) => {
                let path = e
                    .path()
                    .map(Path::to_path_buf)
                    .unwrap_or_else(|| root.to_path_buf());
                failures.push(failure(&path, e.to_string()));
            }
        }
    }
    (paths, failures)
}

fn read_all(paths: &[PathBuf], max_threads: Option<usize>) -> Result<ReadOutcome, String> {
    let read = || -> Vec<Result<AudioFile, FileFailure>> {
        paths
            .par_iter()
            .map(|p| read_audio_file(p).map_err(|reason| failure(p, reason)))
            .collect()
    };
    let results = match max_threads {
        Some(n) => rayon::ThreadPoolBuilder::new()
            .num_threads(n.max(1))
            .build()
            .map_err(|e| e.to_string())?
            .install(read),
        None => read(),
    };
    let mut files = Vec::with_capacity(results.len());
    let mut failures = Vec::new();
    for result in results {
        match result {
            Ok(file) => files.push(file),
            Err(f) => failures.push(f),
        }
    }
    files.sort_by(|a, b| a.path.cmp(&b.path));
    Ok((files, failures))
}

/// Recursively scan `root` for supported audio files and read their embedded metadata in parallel.
#[pyfunction]
#[pyo3(signature = (root, recursive = true, max_threads = None))]
pub fn scan_library(
    py: Python<'_>,
    root: PathBuf,
    recursive: bool,
    max_threads: Option<usize>,
) -> PyResult<ReadOutcome> {
    if !root.is_dir() {
        return Err(PyNotADirectoryError::new_err(format!(
            "not a directory: {}",
            root.display()
        )));
    }
    py.detach(|| -> Result<ReadOutcome, String> {
        let (paths, mut walk_failures) = collect_candidates(&root, recursive);
        let (files, mut read_failures) = read_all(&paths, max_threads)?;
        walk_failures.append(&mut read_failures);
        Ok((files, walk_failures))
    })
    .map_err(PyRuntimeError::new_err::<String>)
}

/// Read embedded metadata for an explicit batch of files.
#[pyfunction]
#[pyo3(signature = (paths, max_threads = None))]
pub fn read_metadata(
    py: Python<'_>,
    paths: Vec<PathBuf>,
    max_threads: Option<usize>,
) -> PyResult<ReadOutcome> {
    py.detach(|| read_all(&paths, max_threads))
        .map_err(PyRuntimeError::new_err::<String>)
}
