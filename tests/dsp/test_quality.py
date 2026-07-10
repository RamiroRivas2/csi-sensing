import numpy as np

from csi.dsp.quality import link_quality

FS = 20.0


def _signal(breathing_amp: float, noise: float, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(int(60 * FS)) / FS
    return breathing_amp * np.sin(2 * np.pi * 0.25 * t) + rng.normal(0, noise, t.shape[0])


def test_strong_breathing_scores_high():
    q = link_quality(_signal(1.0, 0.05), FS)
    assert q.verdict == "excellent"
    assert q.score > 80
    assert q.snr_db > 20


def test_weak_breathing_scores_low():
    q = link_quality(_signal(0.05, 1.0), FS)
    assert q.verdict in ("poor", "fair")
    assert q.score < 40


def test_pure_noise_is_poor():
    rng = np.random.default_rng(1)
    q = link_quality(rng.normal(0, 1.0, int(60 * FS)), FS)
    assert q.verdict == "poor"


def test_ordering_is_monotonic():
    strong = link_quality(_signal(1.0, 0.1), FS)
    medium = link_quality(_signal(0.4, 0.4), FS)
    weak = link_quality(_signal(0.1, 0.8), FS)
    assert strong.score > medium.score > weak.score
