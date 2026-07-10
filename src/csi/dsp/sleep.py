"""Sleep-quality metrics from an overnight CSI session.

These are the objective sleep markers the mood/depression literature keys on
(sleep efficiency, awakenings, restlessness, schedule regularity). They are
WELLBEING INDICATORS: sustained deviations from a personal baseline are worth
noticing, but nothing here diagnoses any condition.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from csi.dsp.breathing import breathing_timeline
from csi.dsp.vitals import activity_timeline


@dataclass
class SleepMetrics:
    duration_h: float
    present_fraction: float  # time someone was in range
    sleep_efficiency: float  # still / present - the classic efficiency proxy
    awakenings: int  # distinct motion bouts of >= 30 s while present
    restlessness: float  # fraction of present time spent moving
    longest_still_h: float  # longest uninterrupted still stretch
    breathing_median_bpm: float | None
    breathing_iqr_bpm: float | None  # overnight rate spread


def sleep_metrics(amp: np.ndarray, fs: float) -> SleepMetrics:
    duration_h = amp.shape[0] / fs / 3600

    times, activity = activity_timeline(amp, fs, window_s=10.0, hop_s=5.0)
    if not activity:
        return SleepMetrics(round(duration_h, 2), 0.0, 0.0, 0, 0.0, 0.0, None, None)

    states = [a.state for a in activity]
    present = [s != "empty" for s in states]
    moving = [s == "moving" for s in states]
    n = len(states)
    present_fraction = sum(present) / n

    present_states = [s for s in states if s != "empty"]
    sleep_efficiency = (
        present_states.count("still") / len(present_states) if present_states else 0.0
    )
    restlessness = (
        sum(m for m, p in zip(moving, present, strict=True) if p) / sum(present)
        if any(present)
        else 0.0
    )

    hop_s = float(times[1] - times[0]) if len(times) > 1 else 5.0
    min_bout_bins = max(1, int(30.0 / hop_s))
    awakenings = 0
    run = 0
    for m in moving:
        run = run + 1 if m else 0
        if run == min_bout_bins:
            awakenings += 1

    longest = run_len = 0
    for s in states:
        run_len = run_len + 1 if s == "still" else 0
        longest = max(longest, run_len)
    longest_still_h = longest * hop_s / 3600

    _, breathing = breathing_timeline(amp, fs)
    bpms = [e.bpm for e in breathing if e.confidence > 0.05]
    breathing_median = round(float(np.median(bpms)), 1) if bpms else None
    breathing_iqr = (
        round(float(np.percentile(bpms, 75) - np.percentile(bpms, 25)), 1) if bpms else None
    )

    return SleepMetrics(
        duration_h=round(duration_h, 2),
        present_fraction=round(present_fraction, 2),
        sleep_efficiency=round(sleep_efficiency, 2),
        awakenings=awakenings,
        restlessness=round(restlessness, 3),
        longest_still_h=round(longest_still_h, 2),
        breathing_median_bpm=breathing_median,
        breathing_iqr_bpm=breathing_iqr,
    )
