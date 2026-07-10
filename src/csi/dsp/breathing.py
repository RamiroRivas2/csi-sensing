"""Breathing-rate extraction from CSI amplitude. Pure signal processing, no ML.

Breathing (chest displacement) modulates multipath coherently across subcarriers at
0.2-0.5 Hz. Pipeline: PCA across subcarriers (when given a matrix) -> zero-phase
band-pass -> Welch PSD -> dominant in-band peak -> breaths per minute.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal

from csi.dsp.filters import bandpass_filter, pca_denoise

DEFAULT_BAND: tuple[float, float] = (0.1, 0.7)  # 6 to 42 breaths/min


@dataclass
class BreathingEstimate:
    bpm: float
    peak_hz: float
    confidence: float  # peak power / total in-band power, in (0, 1]
    freqs: np.ndarray
    psd: np.ndarray


def estimate_breathing_rate(
    x: np.ndarray,
    fs: float,
    band: tuple[float, float] = DEFAULT_BAND,
) -> BreathingEstimate:
    """Estimate breathing rate from a 1D signal or a (T, S) subcarrier matrix.

    Needs at least ~2 breathing cycles in x to resolve a peak; 30 s is comfortable.
    """
    if x.ndim == 2:
        _, components = pca_denoise(x, n_components=1)
        x = components[:, 0]
    elif x.ndim != 1:
        raise ValueError(f"expected 1D signal or (T, S) matrix, got shape {x.shape}")

    low, high = band
    filtered = bandpass_filter(x - x.mean(), low, high, fs)

    # Welch with a segment long enough to resolve ~0.03 Hz spacing at the low end
    nperseg = min(filtered.shape[0], int(fs * 40))
    freqs, psd = signal.welch(filtered, fs=fs, nperseg=nperseg)

    mask = (freqs >= low) & (freqs <= high)
    band_freqs, band_psd = freqs[mask], psd[mask]
    if band_psd.size == 0 or band_psd.sum() <= 0:
        return BreathingEstimate(0.0, 0.0, 0.0, freqs, psd)

    peak_idx = int(np.argmax(band_psd))
    peak_hz = float(band_freqs[peak_idx])
    # parabolic interpolation between bins: Welch resolution alone quantizes to
    # ~2 bpm steps, which is coarser than real breathing-rate changes
    if 0 < peak_idx < band_psd.size - 1:
        left, center, right = band_psd[peak_idx - 1 : peak_idx + 2]
        denom = left - 2 * center + right
        if denom < 0:
            shift = 0.5 * (left - right) / denom
            peak_hz += shift * float(band_freqs[1] - band_freqs[0])
    confidence = float(band_psd[peak_idx] / band_psd.sum())
    return BreathingEstimate(
        bpm=60.0 * peak_hz,
        peak_hz=peak_hz,
        confidence=confidence,
        freqs=freqs,
        psd=psd,
    )


def breathing_timeline(
    x: np.ndarray,
    fs: float,
    window_s: float = 30.0,
    hop_s: float = 5.0,
    band: tuple[float, float] = DEFAULT_BAND,
) -> tuple[np.ndarray, list[BreathingEstimate]]:
    """Sliding breathing estimates over a long recording.

    Returns (window center times in seconds, estimates).
    """
    win = int(window_s * fs)
    hop = max(1, int(hop_s * fs))
    n = x.shape[0]
    if n < win:
        return np.empty(0), []
    starts = np.arange(0, n - win + 1, hop)
    estimates = [estimate_breathing_rate(x[s : s + win], fs, band) for s in starts]
    times = (starts + win / 2) / fs
    return times, estimates
