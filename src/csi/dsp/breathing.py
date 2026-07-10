"""Breathing-rate extraction from CSI amplitude. Pure signal processing, no ML.

Thin domain wrapper over the generalized band-rate estimator in csi.dsp.vitals:
breathing appears as the 0.1-0.7 Hz coherent component of subcarrier amplitude.
"""

from __future__ import annotations

import numpy as np

from csi.dsp.vitals import BREATHING_BAND, RateEstimate, estimate_rate, rate_timeline

DEFAULT_BAND = BREATHING_BAND

# public alias: breathing estimates are plain rate estimates
BreathingEstimate = RateEstimate


def estimate_breathing_rate(
    x: np.ndarray,
    fs: float,
    band: tuple[float, float] = DEFAULT_BAND,
) -> BreathingEstimate:
    """Estimate breathing rate from a 1D signal or a (T, S) subcarrier matrix.

    Needs at least ~2 breathing cycles in x to resolve a peak; 30 s is comfortable.
    """
    return estimate_rate(x, fs, band)


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
    return rate_timeline(x, fs, band, window_s, hop_s)
