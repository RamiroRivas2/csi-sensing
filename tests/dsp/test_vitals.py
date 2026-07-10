import numpy as np
import pytest

from csi.dsp.vitals import (
    activity_timeline,
    classify_activity,
    estimate_heart_rate,
    estimate_rate,
    heart_rate_timeline,
    rate_timeline,
)

FS = 20.0


def _vitals_signal(
    duration_s: float,
    breathing_bpm: float = 15.0,
    heart_bpm: float = 72.0,
    heart_amp: float = 0.15,
    noise: float = 0.15,
    seed: int = 0,
) -> np.ndarray:
    """Breathing + a much weaker cardiac component + white noise."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(duration_s * FS)) / FS
    sig = np.sin(2 * np.pi * (breathing_bpm / 60) * t)
    sig = sig + heart_amp * np.sin(2 * np.pi * (heart_bpm / 60) * t)
    return sig + rng.normal(0, noise, t.shape[0])


# Heart rates chosen NOT to coincide exactly with harmonics of the 15 bpm (0.25 Hz)
# breathing carrier: 60 bpm (1.0 Hz) and 90 bpm (1.5 Hz) sit exactly on the 4th/6th
# breathing harmonic, where notching correctly removes the real signal too - that
# exact-collision case is a documented physical limit, not a recoverable rate.
@pytest.mark.parametrize("heart_bpm", [66.0, 72.0, 84.0, 110.0])
def test_heart_rate_recovered_within_2bpm(heart_bpm: float):
    x = _vitals_signal(120.0, heart_bpm=heart_bpm)
    est = estimate_heart_rate(x, FS)
    assert abs(est.bpm - heart_bpm) <= 2.0


def test_pure_breathing_harmonics_do_not_forge_a_heart_rate():
    """Non-sinusoidal breathing with NO cardiac component must not report a
    confident heart rate from its harmonics (the harmonic-confusion failure mode)."""
    rng = np.random.default_rng(9)
    t = np.arange(int(120 * FS)) / FS
    # triangular breathing at 18 bpm (0.3 Hz): rich harmonics into the 0.8-2.2 Hz band
    breathing = signal_triangle(0.3, t)
    x = breathing + rng.normal(0, 0.05, t.shape[0])
    est = estimate_heart_rate(x, FS)
    # after notching the breathing harmonics, no strong cardiac peak should survive;
    # if a rate is reported at all it must be low-confidence, not a plausible HR
    assert est.confidence < 0.5


def signal_triangle(freq: float, t: np.ndarray) -> np.ndarray:
    from scipy.signal import sawtooth

    return sawtooth(2 * np.pi * freq * t, width=0.5)


def test_heart_rate_from_matrix():
    rng = np.random.default_rng(1)
    base = _vitals_signal(120.0, heart_bpm=72.0)
    weights = rng.uniform(0.5, 1.5, 40)
    matrix = np.outer(base, weights) + rng.normal(0, 0.1, (base.shape[0], 40))
    est = estimate_heart_rate(matrix, FS)
    assert abs(est.bpm - 72.0) <= 2.0


def test_heart_and_breathing_do_not_cross_contaminate():
    x = _vitals_signal(120.0, breathing_bpm=18.0, heart_bpm=66.0)
    breathing = estimate_rate(x, FS, (0.1, 0.7))
    heart = estimate_heart_rate(x, FS)
    assert abs(breathing.bpm - 18.0) <= 1.0
    assert abs(heart.bpm - 66.0) <= 2.0


def test_rate_timeline_shapes():
    x = _vitals_signal(120.0)
    times, estimates = rate_timeline(x, FS, (0.8, 2.2), window_s=20.0, hop_s=5.0)
    assert len(times) == len(estimates) > 0


def test_heart_rate_timeline_recovers_real_rate():
    x = _vitals_signal(120.0, heart_bpm=72.0)
    times, estimates = heart_rate_timeline(x, FS, window_s=20.0, hop_s=5.0)
    assert len(times) == len(estimates) > 0
    assert abs(np.median([e.bpm for e in estimates]) - 72.0) <= 3.0


def test_heart_rate_timeline_resists_pure_breathing_forgery():
    """The per-window timeline must apply the same breathing-harmonic notch as
    estimate_heart_rate, so pure breathing yields only low-confidence windows."""
    rng = np.random.default_rng(9)
    t = np.arange(int(120 * FS)) / FS
    breathing = signal_triangle(0.3, t)
    x = breathing + rng.normal(0, 0.05, t.shape[0])
    times, estimates = heart_rate_timeline(x, FS, window_s=20.0, hop_s=5.0)
    assert len(times) == len(estimates) > 0
    assert np.median([e.confidence for e in estimates]) < 0.5


class TestActivity:
    def test_empty_room_is_white_noise(self):
        rng = np.random.default_rng(2)
        x = rng.normal(0, 1.0, int(60 * FS))
        est = classify_activity(x, FS)
        assert est.state == "empty"

    def test_breathing_person_is_still(self):
        x = _vitals_signal(60.0, noise=0.1)
        est = classify_activity(x, FS)
        assert est.state == "still"
        assert est.presence_score > 0.6

    def test_large_body_motion_is_moving(self):
        rng = np.random.default_rng(3)
        t = np.arange(int(60 * FS)) / FS
        motion = sum(
            np.sin(2 * np.pi * f * t + rng.uniform(0, 2 * np.pi)) for f in (1.1, 1.9, 2.7, 3.6)
        )
        x = _vitals_signal(60.0, noise=0.1) + 2.0 * motion
        est = classify_activity(x, FS)
        assert est.state == "moving"

    def test_activity_timeline_tracks_transition(self):
        rng = np.random.default_rng(4)
        still = _vitals_signal(60.0, noise=0.1, seed=5)
        t = np.arange(int(60 * FS)) / FS
        motion = 2.0 * sum(
            np.sin(2 * np.pi * f * t + rng.uniform(0, 2 * np.pi)) for f in (1.3, 2.4, 3.5)
        )
        x = np.concatenate([still, still + motion])
        times, estimates = activity_timeline(x, FS)
        early = [e.state for e, tt in zip(estimates, times, strict=True) if tt < 50]
        late = [e.state for e, tt in zip(estimates, times, strict=True) if tt > 70]
        assert early.count("still") > len(early) / 2
        assert late.count("moving") > len(late) / 2
