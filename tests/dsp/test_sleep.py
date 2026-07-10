import numpy as np

from csi.dsp.sleep import sleep_metrics

FS = 20.0


def _night(
    hours: float = 0.5,
    awakenings: int = 2,
    bout_s: float = 60.0,
    seed: int = 0,
) -> np.ndarray:
    """Synthetic 'night': breathing sleep with injected motion bouts, as (T, S) matrix."""
    rng = np.random.default_rng(seed)
    n = int(hours * 3600 * FS)
    t = np.arange(n) / FS
    base = np.sin(2 * np.pi * 0.25 * t)  # 15 bpm
    if awakenings:
        for k in range(awakenings):
            start = int((k + 1) * n / (awakenings + 1.5))
            span = int(bout_s * FS)
            tt = np.arange(span) / FS
            base[start : start + span] += 3.0 * (
                np.sin(2 * np.pi * 1.4 * tt) + np.sin(2 * np.pi * 2.9 * tt + 1.0)
            )
    weights = rng.uniform(0.5, 1.5, 30)
    return np.outer(base, weights) + rng.normal(0, 0.1, (n, 30))


def test_quiet_night_metrics():
    m = sleep_metrics(_night(awakenings=0), FS)
    assert m.present_fraction > 0.95
    assert m.sleep_efficiency > 0.9
    assert m.awakenings == 0
    assert m.restlessness < 0.1
    assert m.breathing_median_bpm is not None
    assert abs(m.breathing_median_bpm - 15.0) <= 1.0


def test_awakenings_counted():
    m = sleep_metrics(_night(awakenings=2), FS)
    assert m.awakenings == 2
    assert m.restlessness > 0.02


def test_restless_night_scores_worse_than_quiet():
    quiet = sleep_metrics(_night(awakenings=0, seed=1), FS)
    restless = sleep_metrics(_night(awakenings=4, bout_s=90.0, seed=2), FS)
    assert restless.sleep_efficiency < quiet.sleep_efficiency
    assert restless.restlessness > quiet.restlessness
    assert restless.longest_still_h < quiet.longest_still_h
