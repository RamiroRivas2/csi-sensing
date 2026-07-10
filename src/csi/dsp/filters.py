"""Denoising primitives for CSI amplitude streams.

All filters are zero-phase (sosfiltfilt) because breathing-rate extraction reads
peak positions out of the PSD - phase distortion would silently shift them.
"""

from __future__ import annotations

import numpy as np
from scipy import signal


def butter_bandpass_sos(low_hz: float, high_hz: float, fs: float, order: int = 4) -> np.ndarray:
    """Second-order-sections Butterworth band-pass design."""
    nyq = fs / 2.0
    if not 0 < low_hz < high_hz < nyq:
        raise ValueError(f"need 0 < low ({low_hz}) < high ({high_hz}) < nyquist ({nyq})")
    return signal.butter(order, [low_hz / nyq, high_hz / nyq], btype="band", output="sos")


def bandpass_filter(
    x: np.ndarray, low_hz: float, high_hz: float, fs: float, order: int = 4
) -> np.ndarray:
    """Zero-phase band-pass along the time axis (axis 0)."""
    sos = butter_bandpass_sos(low_hz, high_hz, fs, order)
    return signal.sosfiltfilt(sos, x, axis=0)


def hampel_filter(x: np.ndarray, window_size: int = 11, n_sigmas: float = 3.0) -> np.ndarray:
    """Replace outliers with the local median (1D).

    A sample is an outlier when it deviates from the rolling median by more than
    n_sigmas * 1.4826 * MAD. Robust against the impulsive spikes ESP32 CSI produces.
    """
    if x.ndim != 1:
        raise ValueError("hampel_filter expects a 1D signal")
    if window_size % 2 == 0:
        raise ValueError("window_size must be odd")
    half = window_size // 2
    padded = np.pad(x, half, mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, window_size)
    medians = np.median(windows, axis=1)
    mad = np.median(np.abs(windows - medians[:, None]), axis=1)
    threshold = n_sigmas * 1.4826 * mad
    out = x.copy()
    outliers = np.abs(x - medians) > threshold
    out[outliers] = medians[outliers]
    return out


def pca_denoise(x: np.ndarray, n_components: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Project (T, S) subcarrier matrix onto its top principal components.

    Breathing moves all subcarriers coherently, so the common motion signal
    concentrates in the first few components while per-subcarrier noise does not.

    Returns (reconstructed (T, S), components (T, n_components)).
    """
    if x.ndim != 2:
        raise ValueError("pca_denoise expects (T, S)")
    mean = x.mean(axis=0)
    centered = x - mean
    # SVD-based PCA: T x S with S small (tens of subcarriers), cheap on CPU
    u, s, vt = np.linalg.svd(centered, full_matrices=False)
    k = min(n_components, s.shape[0])
    components = u[:, :k] * s[:k]
    reconstructed = components @ vt[:k] + mean
    return reconstructed, components
