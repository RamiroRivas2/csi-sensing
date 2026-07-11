"""Link quality: how well this node placement can see breathing.

The commercial setup wizard boils down to this number - move the nodes until the
score goes green. SNR here is the breathing-band spectral peak against the noise
floor measured where no physiological signal lives (above the motion band's core).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal

from csi.dsp.vitals import BREATHING_BAND, _first_component


@dataclass
class LinkQuality:
    snr_db: float  # breathing peak vs noise floor
    score: int  # 0-100
    verdict: str  # "excellent" | "good" | "fair" | "poor" | "unknown"


VERDICTS: tuple[tuple[float, str], ...] = (
    (20.0, "excellent"),
    (12.0, "good"),
    (6.0, "fair"),
)


def link_quality(x: np.ndarray, fs: float) -> LinkQuality:
    """Score a window (>= ~30 s recommended) of CSI amplitude for breathing visibility."""
    comp = _first_component(x)
    comp = comp - comp.mean()
    freqs, psd = signal.welch(comp, fs=fs, nperseg=min(comp.shape[0], int(fs * 40)))

    band = (freqs >= BREATHING_BAND[0]) & (freqs <= BREATHING_BAND[1])
    # noise floor: median PSD between 3 Hz and 0.9 x nyquist, where neither
    # breathing nor its harmonics nor typical body sway live
    floor_mask = (freqs >= 3.0) & (freqs <= 0.9 * fs / 2)
    if not band.any() or not floor_mask.any():
        # too little data or too low a sample rate (fs <= ~6.7 Hz leaves no
        # spectrum above 3 Hz) to measure the noise floor: the link is not
        # known to be poor, it is unmeasurable - don't send the setup wizard
        # chasing a placement problem that is actually a frame-rate problem
        return LinkQuality(0.0, 0, "unknown")

    peak = float(psd[band].max())
    floor = float(np.median(psd[floor_mask])) or 1e-12
    snr_db = 10 * np.log10(peak / floor) if peak > 0 else 0.0

    score = int(np.clip((snr_db - 3.0) / (26.0 - 3.0) * 100, 0, 100))
    verdict = "poor"
    for threshold, name in VERDICTS:
        if snr_db >= threshold:
            verdict = name
            break
    return LinkQuality(snr_db=round(float(snr_db), 1), score=score, verdict=verdict)
