"""Fall detection from CSI amplitude. Research-grade heuristic, NOT a safety device.

A fall has a characteristic signature: a short, violent broadband motion burst
(the body dropping) followed by sustained stillness (the person on the floor).
This module detects that burst-then-stillness pattern by tracking the motion-band
energy envelope against the session's own baseline.

Honest limits, until an ML classifier is trained on labeled falls (UT-HAR has a
fall class; camera-supervised labeling of own data comes later):
- a violent sit-down or a dropped heavy object can look like a fall (severity helps)
- a fall followed by immediate recovery (getting right back up) is NOT flagged,
  by design - the dangerous case is the person who stays down
- per-room burst_z tuning will be needed on real hardware
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from csi.dsp.filters import bandpass_filter
from csi.dsp.vitals import MOTION_BAND, _first_component


@dataclass
class FallEvent:
    t: float  # seconds from start of signal, at the burst peak
    severity: float  # burst peak as a robust z-score vs session baseline
    stillness_after: float  # fraction of the post-burst window that is still, 0..1
    confidence: float  # severity margin x stillness, squashed to 0..1


def motion_envelope(x: np.ndarray, fs: float, hop_s: float = 0.25) -> tuple[np.ndarray, np.ndarray]:
    """RMS envelope of the motion band (times, envelope values)."""
    comp = _first_component(x)
    high = min(MOTION_BAND[1], 0.45 * fs)
    motion = bandpass_filter(comp - comp.mean(), MOTION_BAND[0], high, fs)
    hop = max(1, int(hop_s * fs))
    win = 2 * hop
    n = motion.shape[0]
    if n < win:
        return np.empty(0), np.empty(0)
    starts = np.arange(0, n - win + 1, hop)
    env = np.array([np.sqrt(np.mean(motion[s : s + win] ** 2)) for s in starts])
    times = (starts + win / 2) / fs
    return times, env


def detect_falls(
    x: np.ndarray,
    fs: float,
    burst_z: float = 6.0,
    max_burst_s: float = 6.0,
    still_s: float = 8.0,
    still_z: float = 2.0,
    min_stillness: float = 0.8,
) -> list[FallEvent]:
    """Detect burst-then-stillness events.

    burst_z: how many robust standard deviations above baseline the burst must peak.
    max_burst_s: longer above-threshold episodes are activity (walking, exercise),
      not the 1-2 s impact of a fall. Measured on the smoothed envelope, which the
      RMS window and zero-phase filter smear 2-3 s beyond the physical impact at
      good SNR, so the physical cutoff is roughly max_burst_s minus that smear.
    still_s: how long after the burst the signal must stay near baseline.
    min_stillness: fraction of the post-burst window required below still_z.
    """
    times, env = motion_envelope(x, fs)
    if env.size < 10:
        return []

    median = float(np.median(env))
    mad = float(np.median(np.abs(env - median))) or 1e-9
    z = (env - median) / (1.4826 * mad)

    hop_s = float(times[1] - times[0]) if times.size > 1 else 0.25
    still_bins = max(1, int(still_s / hop_s))

    events: list[FallEvent] = []
    above = z > burst_z
    i = 0
    while i < z.size:
        if not above[i]:
            i += 1
            continue
        # group the contiguous burst
        j = i
        while j < z.size and above[j]:
            j += 1
        if (j - i) * hop_s > max_burst_s:
            i = j  # sustained motion episode, not an impact
            continue
        peak_idx = i + int(np.argmax(z[i:j]))
        after = z[j : j + still_bins]
        if after.size >= still_bins:  # enough signal after the burst to judge
            stillness = float(np.mean(after < still_z))
            if stillness >= min_stillness:
                severity = float(z[peak_idx])
                confidence = float(min(1.0, (severity / burst_z - 1.0)) * stillness)
                events.append(
                    FallEvent(
                        t=float(times[peak_idx]),
                        severity=round(severity, 1),
                        stillness_after=round(stillness, 2),
                        confidence=round(max(0.0, min(1.0, confidence)), 2),
                    )
                )
                i = j + still_bins  # skip the accepted event's stillness window
                continue
        # rejected candidate: resume right after the burst, so a real fall inside
        # what would have been this candidate's stillness window is still scanned
        i = j
    return events
