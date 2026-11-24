//! Native systems layer for Proteus, exposed to Python as `proteus_music._native`.

mod decode;
mod fingerprint;
mod metadata;
mod scan;

use pyo3::prelude::*;

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<metadata::AudioFile>()?;
    m.add_class::<metadata::FileFailure>()?;
    m.add_class::<decode::DecodedAudio>()?;
    m.add_function(wrap_pyfunction!(decode::decode_audio, m)?)?;
    m.add_function(wrap_pyfunction!(fingerprint::fingerprint_audio, m)?)?;
    m.add_function(wrap_pyfunction!(scan::scan_library, m)?)?;
    m.add_function(wrap_pyfunction!(scan::read_metadata, m)?)?;
    Ok(())
}
