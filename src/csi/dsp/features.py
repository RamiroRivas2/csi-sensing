"""Feature extraction for classical ML on CSI windows."""

from __future__ import annotations

import numpy as np
from scipy import signal, stats

DEFAULT_BANDS: tuple[tuple[float, float], ...] = (
    (0.1, 0.7),  # breathing
    (0.7, 2.0),  # slow body motion
    (2.0, 5.0),  # walking / limb motion
    (5.0, 10.0),  # fast motion
)


def dominant_frequency(x: np.ndarray, fs: float) -> float:
    """Frequency (Hz) of the largest non-DC spectral peak of a 1D signal."""
    x = x - x.mean()
    freqs = np.fft.rfftfreq(x.shape[0], d=1 / fs)
    spectrum = np.abs(np.fft.rfft(x))
    spectrum[0] = 0.0
    return float(freqs[int(np.argmax(spectrum))])


def band_energies(
    x: np.ndarray, fs: float, bands: tuple[tuple[float, float], ...] = DEFAULT_BANDS
) -> np.ndarray:
    """Total PSD energy per frequency band, clipped to what fs can represent."""
    freqs, psd = signal.welch(x - x.mean(), fs=fs, nperseg=min(256, x.shape[0]))
    out = np.empty(len(bands))
    for i, (lo, hi) in enumerate(bands):
        mask = (freqs >= lo) & (freqs < hi)
        out[i] = psd[mask].sum() if mask.any() else 0.0
    return out


def statistical_moments(x: np.ndarray) -> np.ndarray:
    """mean, std, skew, kurtosis, iqr, mad of a 1D signal."""
    return np.array(
        [
            x.mean(),
            x.std(),
            stats.skew(x),
            stats.kurtosis(x),
            stats.iqr(x),
            stats.median_abs_deviation(x),
        ]
    )


def spectrogram(
    x: np.ndarray, fs: float, nperseg: int = 64
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Thin wrapper over scipy.signal.spectrogram (freqs, times, Sxx)."""
    nperseg = min(nperseg, x.shape[0])
    return signal.spectrogram(x - x.mean(), fs=fs, nperseg=nperseg)


def feature_names(bands: tuple[tuple[float, float], ...] = DEFAULT_BANDS) -> list[str]:
    return [
        "dominant_freq",
        *[f"band_{lo}_{hi}" for lo, hi in bands],
        "mean",
        "std",
        "skew",
        "kurtosis",
        "iqr",
        "mad",
    ]


def extract_feature_vector(
    window: np.ndarray, fs: float, bands: tuple[tuple[float, float], ...] = DEFAULT_BANDS
) -> np.ndarray:
    """Feature vector for one 1D window. Order matches feature_names()."""
    if window.ndim != 1:
        raise ValueError("extract_feature_vector expects a 1D window")
    return np.concatenate(
        [
            [dominant_frequency(window, fs)],
            band_energies(window, fs, bands),
            statistical_moments(window),
        ]
    )
