import numpy as np
import pytest

from csi.dsp.breathing import breathing_timeline, estimate_breathing_rate

FS = 20.0


def _breathing_signal(
    bpm: float, duration_s: float, snr_db: float = 0.0, seed: int = 0
) -> np.ndarray:
    """Sinusoid at the breathing rate buried in white noise at the given SNR."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(duration_s * FS)) / FS
    sig = np.sin(2 * np.pi * (bpm / 60.0) * t)
    noise_power = np.mean(sig**2) / (10 ** (snr_db / 10))
    return sig + rng.normal(0, np.sqrt(noise_power), t.shape[0])


@pytest.mark.parametrize("bpm", [9.0, 15.0, 21.0, 30.0, 36.0])
def test_rate_recovered_within_1bpm_at_0db_snr(bpm: float):
    x = _breathing_signal(bpm, duration_s=120.0)
    est = estimate_breathing_rate(x, FS)
    assert abs(est.bpm - bpm) <= 1.0
    assert est.confidence > 0.05


def test_multichannel_matrix_input():
    rng = np.random.default_rng(2)
    x = _breathing_signal(15.0, 120.0)
    weights = rng.uniform(0.5, 1.5, 40)
    matrix = np.outer(x, weights) + rng.normal(0, 0.3, (x.shape[0], 40))
    est = estimate_breathing_rate(matrix, FS)
    assert abs(est.bpm - 15.0) <= 1.0


def test_timeline_tracks_rate_step():
    first = _breathing_signal(12.0, 120.0, seed=3)
    second = _breathing_signal(24.0, 120.0, seed=4)
    x = np.concatenate([first, second])
    times, estimates = breathing_timeline(x, FS, window_s=30.0, hop_s=5.0)
    assert len(estimates) == len(times) > 0
    early = [e.bpm for e, t in zip(estimates, times, strict=True) if t < 100]
    late = [e.bpm for e, t in zip(estimates, times, strict=True) if t > 140]
    assert abs(np.median(early) - 12.0) <= 1.5
    assert abs(np.median(late) - 24.0) <= 1.5


def test_short_input_returns_empty_timeline():
    times, estimates = breathing_timeline(np.zeros(10), FS)
    assert times.size == 0 and estimates == []
