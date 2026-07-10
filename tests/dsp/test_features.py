import numpy as np

from csi.dsp.features import (
    dominant_frequency,
    extract_feature_vector,
    feature_names,
    statistical_moments,
)

FS = 20.0


def test_dominant_frequency_of_pure_tone():
    t = np.arange(int(60 * FS)) / FS
    x = np.sin(2 * np.pi * 0.5 * t)
    resolution = FS / x.shape[0]
    assert abs(dominant_frequency(x, FS) - 0.5) <= resolution


def test_moments_of_known_gaussian():
    rng = np.random.default_rng(0)
    x = rng.normal(3.0, 2.0, 200_000)
    mean, std, skew, kurt, _, _ = statistical_moments(x)
    assert abs(mean - 3.0) < 0.05
    assert abs(std - 2.0) < 0.05
    assert abs(skew) < 0.05
    assert abs(kurt) < 0.1


def test_feature_vector_matches_names():
    t = np.arange(int(10 * FS)) / FS
    x = np.sin(2 * np.pi * 0.3 * t)
    vec = extract_feature_vector(x, FS)
    assert vec.shape[0] == len(feature_names())
    assert np.all(np.isfinite(vec))
