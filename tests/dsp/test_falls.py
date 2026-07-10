import numpy as np

from csi.dsp.falls import detect_falls

FS = 20.0


def _breathing(duration_s: float, seed: int = 0, noise: float = 0.1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(int(duration_s * FS)) / FS
    return np.sin(2 * np.pi * 0.25 * t) + rng.normal(0, noise, t.shape[0])


def _burst(duration_s: float, amplitude: float = 10.0, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(0, amplitude, int(duration_s * FS))


def test_fall_detected_at_right_time():
    x = _breathing(120.0)
    fall_at = 60.0
    i = int(fall_at * FS)
    x[i : i + int(1.5 * FS)] += _burst(1.5)
    events = detect_falls(x, FS)
    assert len(events) == 1
    assert abs(events[0].t - (fall_at + 0.75)) < 3.0
    assert events[0].confidence > 0.3


def test_no_fall_in_quiet_breathing():
    assert detect_falls(_breathing(120.0), FS) == []


def test_continuous_walking_is_not_a_fall():
    rng = np.random.default_rng(2)
    t = np.arange(int(120 * FS)) / FS
    walking = sum(np.sin(2 * np.pi * f * t + rng.uniform(0, 2 * np.pi)) for f in (1.2, 2.1, 3.4))
    x = 2.0 * walking + rng.normal(0, 0.2, t.shape[0])
    assert detect_falls(x, FS) == []


def test_burst_followed_by_more_motion_is_not_a_fall():
    rng = np.random.default_rng(3)
    x = _breathing(120.0)
    i = int(60.0 * FS)
    x[i : i + int(1.5 * FS)] += _burst(1.5)
    # person keeps moving vigorously afterwards - they got back up
    t_after = np.arange(int(30 * FS)) / FS
    x[i + int(2 * FS) : i + int(2 * FS) + t_after.shape[0]] += 3.0 * np.sin(
        2 * np.pi * 2.0 * t_after
    ) + rng.normal(0, 1.0, t_after.shape[0])
    assert detect_falls(x, FS) == []


def test_fall_shortly_after_rejected_blip_is_still_detected():
    # a dropped object at t=60 is a burst candidate that fails the stillness
    # check (the real fall lands inside its window); the scan must resume right
    # after the blip, not skip past the fall
    x = _breathing(120.0)
    i = int(60.0 * FS)
    x[i : i + int(0.5 * FS)] += _burst(0.5, seed=6)
    j = int(64.0 * FS)
    x[j : j + int(1.5 * FS)] += _burst(1.5, seed=7)
    events = detect_falls(x, FS)
    assert len(events) == 1
    assert abs(events[0].t - 64.75) < 3.0


def test_two_second_impact_is_still_a_fall():
    # envelope smoothing and filtfilt ringing smear the above-threshold group
    # well past the physical impact; a 2 s impact must not be rejected as
    # sustained motion
    x = _breathing(120.0)
    i = int(60.0 * FS)
    x[i : i + int(2.0 * FS)] += _burst(2.0)
    events = detect_falls(x, FS)
    assert len(events) == 1


def test_two_separated_falls():
    x = _breathing(240.0)
    for fall_at, seed in ((60.0, 4), (180.0, 5)):
        i = int(fall_at * FS)
        x[i : i + int(1.5 * FS)] += _burst(1.5, seed=seed)
    events = detect_falls(x, FS)
    assert len(events) == 2
