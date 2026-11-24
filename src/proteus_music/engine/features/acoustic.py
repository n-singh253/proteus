"""Conventional acoustic features from a mono excerpt (CPU-only, NumPy).

Features (in ``FEATURE_NAMES`` order), all from frames of about 46 ms with 50% overlap. Spectral
statistics use only 0-11.025 kHz, the band every common sample rate shares, so a file and its
resampled copy describe alike:

- ``loudness_db``: RMS level of the excerpt in dBFS (not LUFS; no K-weighting).
- ``loudness_range_db``: 95th minus 10th percentile of frame RMS levels (non-silent frames).
- ``centroid_khz`` / ``centroid_std_khz``: spectral centroid mean and spread (brightness).
- ``bandwidth_khz``: spectral bandwidth around the centroid.
- ``rolloff_khz``: frequency below which 85% of spectral energy lies.
- ``flatness``: spectral flatness (0 tonal .. 1 noise-like).
- ``mfcc_1`` .. ``mfcc_13``: mean mel-frequency cepstral coefficients (timbre), from 40 mel
  bands between 30 Hz and 11.025 kHz so files at different sample rates are comparable.

Tempo comes from the autocorrelation of a mel spectral-flux onset envelope. It is reported only
when onsets are pronounced and the autocorrelation peak is strong; otherwise it is missing,
never guessed (a steady tone or noise has no tempo).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.fft import dct

EXTRACTOR_VERSION = "acoustic-v1"
FEATURE_NAMES = (
    "loudness_db", "loudness_range_db", "centroid_khz", "centroid_std_khz", "bandwidth_khz",
    "rolloff_khz", "flatness", *(f"mfcc_{i}" for i in range(1, 14)),
)  # fmt: skip
TEMPO_RANGE_BPM = (50.0, 220.0)
TEMPO_MIN_CONFIDENCE = 0.3
# Onset strength (std of spectral flux, dB per mel band) below which there are no real onsets.
TEMPO_MIN_ONSET_DB = 0.5
MIN_SECONDS = 3.0
_SILENCE_RMS = 1e-4
_MEL_BANDS, _MEL_FMIN, _MEL_FMAX = 40, 30.0, 11025.0


@dataclass(frozen=True)
class AcousticFeatures:
    values: tuple[float, ...]  # Raw values in FEATURE_NAMES order.
    tempo_bpm: float | None
    tempo_confidence: float


class UnanalyzableAudio(ValueError):
    """The excerpt is too short or silent to describe."""


def _frames(x: np.ndarray, size: int, hop: int) -> np.ndarray:
    count = 1 + (len(x) - size) // hop
    index = np.arange(size)[None, :] + hop * np.arange(count)[:, None]
    return x[index]


def _mel_filters(sample_rate: int, n_fft: int) -> np.ndarray:
    def mel(f):
        return 2595.0 * np.log10(1.0 + f / 700.0)

    top = min(_MEL_FMAX, sample_rate / 2)
    edges = 700.0 * (10 ** (np.linspace(mel(_MEL_FMIN), mel(top), _MEL_BANDS + 2) / 2595.0) - 1)
    freqs = np.fft.rfftfreq(n_fft, 1 / sample_rate)
    low, mid, high = edges[:-2, None], edges[1:-1, None], edges[2:, None]
    rising = (freqs - low) / (mid - low)
    falling = (high - freqs) / (high - mid)
    return np.clip(np.minimum(rising, falling), 0, None)


def _tempo(mel_db: np.ndarray, hop_seconds: float) -> tuple[float | None, float]:
    flux = np.clip(np.diff(mel_db, axis=0), 0, None).sum(axis=1)
    if len(flux) < 8 or flux.std() / _MEL_BANDS < TEMPO_MIN_ONSET_DB:
        return None, 0.0
    flux = flux - flux.mean()
    acf = np.correlate(flux, flux, mode="full")[len(flux) - 1 :]
    acf = acf / acf[0]
    lags = np.arange(len(acf))
    bpm = np.divide(60.0, lags * hop_seconds, out=np.zeros(len(acf)), where=lags > 0)
    valid = (bpm >= TEMPO_RANGE_BPM[0]) & (bpm <= TEMPO_RANGE_BPM[1])
    if not valid.any():
        return None, 0.0
    # A log-normal prior around 120 BPM resolves octave ambiguity (half/double tempo).
    prior = np.exp(-0.5 * (np.log2(np.where(valid, bpm, 1) / 120.0) / 1.0) ** 2)
    weighted = np.where(valid, acf * prior, -np.inf)
    lag = int(np.argmax(weighted))
    confidence = float(acf[lag])
    if confidence < TEMPO_MIN_CONFIDENCE:
        return None, confidence
    return float(bpm[lag]), confidence


def extract(samples: np.ndarray, sample_rate: int) -> AcousticFeatures:
    """Describe a mono float excerpt; raises ``UnanalyzableAudio`` when it cannot."""
    x = np.asarray(samples, dtype=np.float64)
    if len(x) < MIN_SECONDS * sample_rate:
        raise UnanalyzableAudio(f"excerpt shorter than {MIN_SECONDS:.0f} s")
    rms = float(np.sqrt(np.mean(x**2)))
    if rms < _SILENCE_RMS:
        raise UnanalyzableAudio("silent excerpt")
    n_fft = 1 << int(np.ceil(np.log2(0.046 * sample_rate)))
    hop = n_fft // 2
    frames = _frames(x, n_fft, hop)
    spectrum = np.abs(np.fft.rfft(frames * np.hanning(n_fft), axis=1)) ** 2
    all_freqs = np.fft.rfftfreq(n_fft, 1 / sample_rate)
    band = all_freqs <= _MEL_FMAX
    power, freqs = spectrum[:, band], all_freqs[band] / 1000.0
    total = power.sum(axis=1) + 1e-12
    centroid = (power * freqs).sum(axis=1) / total
    bandwidth = np.sqrt((power * (freqs - centroid[:, None]) ** 2).sum(axis=1) / total)
    rolloff = freqs[np.argmax(np.cumsum(power, axis=1) >= 0.85 * total[:, None], axis=1)]
    flatness = np.exp(np.mean(np.log(power + 1e-12), axis=1)) / (power.mean(axis=1) + 1e-12)
    frame_db = 10 * np.log10(np.mean(frames**2, axis=1) + 1e-12)
    voiced = frame_db[frame_db > -70]
    spread = np.percentile(voiced, 95) - np.percentile(voiced, 10) if len(voiced) else 0.0
    mel_db = 10 * np.log10(spectrum @ _mel_filters(sample_rate, n_fft).T + 1e-10)
    mfcc = dct(mel_db, type=2, norm="ortho", axis=1)[:, 1:14].mean(axis=0)
    tempo, confidence = _tempo(mel_db, hop / sample_rate)
    values = (
        20 * np.log10(rms), spread, centroid.mean(), centroid.std(), bandwidth.mean(),
        rolloff.mean(), flatness.mean(), *mfcc,
    )  # fmt: skip
    return AcousticFeatures(tuple(float(v) for v in values), tempo, confidence)


# Tempo is encoded as a soft histogram over log-spaced bins (12 per octave), so nearby tempi
# are similar and the encoding stays meaningful under cosine similarity.
_TEMPO_BINS = np.log2(TEMPO_RANGE_BPM[0]) + np.arange(0, np.log2(220 / 50) + 1e-9, 1 / 12)


def acoustic_block(values: list[tuple[float, ...] | None]) -> np.ndarray | None:
    """Rows z-scored across tracks with audio, then L2-normalized; absent rows are zero.

    Standardizing makes loudness in dB, frequencies in kHz and MFCCs comparable. With fewer
    than two analyzed tracks there is nothing to compare, so the block is omitted.
    """
    present = [i for i, v in enumerate(values) if v is not None]
    if len(present) < 2:
        return None
    raw = np.array([values[i] for i in present])
    spread = raw.std(axis=0)
    z = (raw - raw.mean(axis=0)) / np.where(spread > 0, spread, 1.0)
    norms = np.linalg.norm(z, axis=1, keepdims=True)
    block = np.zeros((len(values), raw.shape[1]))
    block[present] = np.divide(z, norms, out=np.zeros_like(z), where=norms > 0)
    return block


def tempo_block(tempi: list[float | None]) -> np.ndarray | None:
    present = [i for i, t in enumerate(tempi) if t]
    if len(present) < 2:
        return None
    block = np.zeros((len(tempi), len(_TEMPO_BINS)))
    for i in present:
        row = np.exp(-0.5 * ((np.log2(tempi[i]) - _TEMPO_BINS) * 12) ** 2)
        block[i] = row / np.linalg.norm(row)
    return block


def tempo_bin_names() -> list[str]:
    return [f"{2**b:.0f}bpm" for b in _TEMPO_BINS]
