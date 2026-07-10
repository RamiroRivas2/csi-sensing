"""Vital-sign and activity estimation from CSI amplitude. Pure signal processing.

Periodic chest motion modulates multipath coherently across subcarriers:
breathing at 0.1-0.7 Hz, and the far smaller cardiac impulse at 0.8-2.2 Hz.
Heart rate is EXPERIMENTAL on ESP32 hardware - it needs a stationary subject and
good placement, and its confidence should gate any display. What this module can
never do is assess sinus rhythm: that is electrical (ECG) territory, while CSI
only sees mechanical chest motion.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy import signal

from csi.dsp.filters import bandpass_filter, pca_denoise

BREATHING_BAND: tuple[float, float] = (0.1, 0.7)  # 6-42 breaths/min
HEART_BAND: tuple[float, float] = (0.8, 2.2)  # 48-132 beats/min
MOTION_BAND: tuple[float, float] = (0.7, 5.0)  # limb/body movement


@dataclass
class RateEstimate:
    bpm: float
    peak_hz: float
    confidence: float  # peak power / total in-band power, in (0, 1]
    freqs: np.ndarray
    psd: np.ndarray


def _first_component(x: np.ndarray) -> np.ndarray:
    """Collapse a (T, S) subcarrier matrix to its dominant coherent motion signal."""
    if x.ndim == 2:
        _, components = pca_denoise(x, n_components=1)
        return components[:, 0]
    if x.ndim != 1:
        raise ValueError(f"expected 1D signal or (T, S) matrix, got shape {x.shape}")
    return x


def estimate_rate(
    x: np.ndarray,
    fs: float,
    band: tuple[float, float],
    exclude_hz: tuple[float, ...] = (),
) -> RateEstimate:
    """Dominant periodic rate inside a frequency band, as cycles per minute.

    Band-pass -> Welch PSD -> in-band peak with parabolic interpolation between
    bins (raw Welch resolution quantizes rates to steps coarser than real
    physiological variation).

    ``exclude_hz`` lists frequencies whose nearest PSD bins are barred from the
    peak *search* while still counting toward total in-band power, so residual
    energy there (e.g. imperfectly notched breathing harmonics) can neither win
    the peak nor inflate confidence.

    Note on ``confidence`` (peak power / total in-band power): it is a *relative*
    metric, comparable only across windows of the same length. The Welch segment
    length is min(len, fs*40), so for windows shorter than 40 s the in-band bin
    count changes with duration and the ratio is not comparable across different
    window_s settings. Use it to rank windows of one timeline, not as an absolute
    cross-configuration score.
    """
    x = _first_component(x)
    low, high = band
    filtered = bandpass_filter(x - x.mean(), low, high, fs)

    nperseg = min(filtered.shape[0], int(fs * 40))
    freqs, psd = signal.welch(filtered, fs=fs, nperseg=nperseg)

    mask = (freqs >= low) & (freqs <= high)
    band_freqs, band_psd = freqs[mask], psd[mask]
    if band_psd.size == 0 or band_psd.sum() <= 0:
        return RateEstimate(0.0, 0.0, 0.0, freqs, psd)

    searchable = np.ones(band_freqs.size, dtype=bool)
    if exclude_hz:
        df = float(freqs[1] - freqs[0])
        for fe in exclude_hz:
            searchable &= np.abs(band_freqs - fe) > 0.6 * df
    if not searchable.any():
        return RateEstimate(0.0, 0.0, 0.0, freqs, psd)

    peak_idx = int(np.argmax(np.where(searchable, band_psd, -np.inf)))
    peak_hz = float(band_freqs[peak_idx])
    if 0 < peak_idx < band_psd.size - 1:
        left, center, right = band_psd[peak_idx - 1 : peak_idx + 2]
        denom = left - 2 * center + right
        if denom < 0:
            shift = 0.5 * (left - right) / denom
            peak_hz += shift * float(band_freqs[1] - band_freqs[0])
    confidence = float(band_psd[peak_idx] / band_psd.sum())
    return RateEstimate(
        bpm=60.0 * peak_hz, peak_hz=peak_hz, confidence=confidence, freqs=freqs, psd=psd
    )


def _band_harmonics(
    fundamental_hz: float, band: tuple[float, float], fs: float
) -> tuple[float, ...]:
    """Every harmonic k*fundamental inside ``band`` and below Nyquist."""
    if fundamental_hz <= 0:
        return ()
    low, high = band
    k_lo = max(1, int(np.ceil(low / fundamental_hz)))
    k_hi = int(np.floor(high / fundamental_hz))
    return tuple(
        f
        for f in (k * fundamental_hz for k in range(k_lo, k_hi + 1))
        if low <= f <= high and 0 < f < fs / 2
    )


def _notch_harmonics(
    x: np.ndarray, fs: float, fundamental_hz: float, band: tuple[float, float], q: float = 12.0
) -> np.ndarray:
    """Remove every harmonic of ``fundamental_hz`` inside ``band`` with IIR notches."""
    out = x
    for f in _band_harmonics(fundamental_hz, band, fs):
        b, a = signal.iirnotch(f, q, fs)
        out = signal.filtfilt(b, a, out)
    return out


def estimate_heart_rate(
    x: np.ndarray, fs: float, band: tuple[float, float] = HEART_BAND
) -> RateEstimate:
    """EXPERIMENTAL heart-rate estimate for a stationary subject.

    KNOWN FAILURE MODE (harmonic confusion): breathing is non-sinusoidal, so the
    3rd-5th harmonics of a normal breathing rate (12-18 bpm) land inside the heart
    band (0.8-2.2 Hz). Without suppression, pure breathing can masquerade as a
    confident, physiologically plausible heart rate. To guard against this we first
    estimate the breathing fundamental, notch its in-band harmonics out of the
    signal, and additionally bar the harmonic frequencies from the heart-band peak
    search (on short windows the IIR notch transient does not fully settle, so a
    residual harmonic shoulder could otherwise still win). This reduces but does
    not eliminate the confusion; any heart-rate accuracy claim still requires a
    contact-ECG reference. A true heart rate landing exactly on a breathing
    harmonic is unrecoverable by design. The cardiac signal is ~an order of
    magnitude weaker than breathing, so treat low confidence as "no reading",
    not as a rate.
    """
    sig = _first_component(x)
    breathing = estimate_rate(sig, fs, BREATHING_BAND)
    harmonics = _band_harmonics(breathing.peak_hz, band, fs)
    if harmonics:
        sig = _notch_harmonics(sig, fs, breathing.peak_hz, band)
    return estimate_rate(sig, fs, band, exclude_hz=harmonics)


def _timeline(
    x: np.ndarray,
    fs: float,
    window_s: float,
    hop_s: float,
    estimate_fn: Callable[[np.ndarray], RateEstimate],
) -> tuple[np.ndarray, list[RateEstimate]]:
    x = _first_component(x)
    win = int(window_s * fs)
    hop = max(1, int(hop_s * fs))
    n = x.shape[0]
    if n < win:
        return np.empty(0), []
    starts = np.arange(0, n - win + 1, hop)
    estimates = [estimate_fn(x[s : s + win]) for s in starts]
    times = (starts + win / 2) / fs
    return times, estimates


def rate_timeline(
    x: np.ndarray,
    fs: float,
    band: tuple[float, float],
    window_s: float,
    hop_s: float,
) -> tuple[np.ndarray, list[RateEstimate]]:
    """Sliding rate estimates over a long recording (window center times, estimates)."""
    return _timeline(x, fs, window_s, hop_s, lambda w: estimate_rate(w, fs, band))


def heart_rate_timeline(
    x: np.ndarray,
    fs: float,
    window_s: float,
    hop_s: float,
) -> tuple[np.ndarray, list[RateEstimate]]:
    """Sliding heart-rate estimates with the breathing-harmonic notch applied
    per window (window center times, estimates).

    Use this instead of ``rate_timeline(x, fs, HEART_BAND, ...)`` for cardiac
    timelines: the plain band search is vulnerable to harmonic confusion from
    non-sinusoidal breathing (see ``estimate_heart_rate``).
    """
    return _timeline(x, fs, window_s, hop_s, lambda w: estimate_heart_rate(w, fs))


@dataclass
class ActivityEstimate:
    presence_score: float  # coherent low-frequency energy fraction, 0..1
    motion_score: float  # motion-band share of human-band energy, 0..1
    state: str  # "empty" | "still" | "moving"


def classify_activity(
    x: np.ndarray,
    fs: float,
    presence_threshold: float = 0.45,
    motion_threshold: float = 0.4,
) -> ActivityEstimate:
    """Coarse room-state classification from spectral energy distribution.

    A person present concentrates energy below ~2.5 Hz (breathing, body sway);
    an empty room's noise spreads flat across the spectrum; large body motion
    fills the 0.7-5 Hz band.
    """
    comp = _first_component(x)
    comp = comp - comp.mean()
    nyq = fs / 2
    freqs, psd = signal.welch(comp, fs=fs, nperseg=min(comp.shape[0], int(fs * 20)))

    def band_energy(lo: float, hi: float) -> float:
        m = (freqs >= lo) & (freqs < min(hi, nyq))
        return float(psd[m].sum())

    human = band_energy(0.1, 2.5)
    broad = band_energy(0.1, min(8.0, nyq * 0.9))
    motion = band_energy(*MOTION_BAND)
    low_all = band_energy(0.1, MOTION_BAND[1])

    presence_score = human / broad if broad > 0 else 0.0
    motion_score = motion / low_all if low_all > 0 else 0.0

    if presence_score < presence_threshold:
        state = "empty"
    elif motion_score > motion_threshold:
        state = "moving"
    else:
        state = "still"
    return ActivityEstimate(
        presence_score=round(presence_score, 3),
        motion_score=round(motion_score, 3),
        state=state,
    )


def activity_timeline(
    x: np.ndarray, fs: float, window_s: float = 10.0, hop_s: float = 5.0
) -> tuple[np.ndarray, list[ActivityEstimate]]:
    comp = _first_component(x)
    win = int(window_s * fs)
    hop = max(1, int(hop_s * fs))
    n = comp.shape[0]
    if n < win:
        return np.empty(0), []
    starts = np.arange(0, n - win + 1, hop)
    estimates = [classify_activity(comp[s : s + win], fs) for s in starts]
    times = (starts + win / 2) / fs
    return times, estimates
