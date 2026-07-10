import numpy as np
import pytest

from csi.dsp.filters import bandpass_filter, hampel_filter, pca_denoise

FS = 20.0


def _tone(freq: float, duration_s: float = 60.0, fs: float = FS) -> np.ndarray:
    t = np.arange(int(duration_s * fs)) / fs
    return np.sin(2 * np.pi * freq * t)


def _power(x: np.ndarray) -> float:
    return float(np.mean(x**2))


class TestBandpass:
    def test_passband_retained_within_1db(self):
        x = _tone(0.3)
        y = bandpass_filter(x, 0.1, 0.7, FS)
        # ignore filter edge transients
        core = slice(200, -200)
        ratio_db = 10 * np.log10(_power(y[core]) / _power(x[core]))
        assert abs(ratio_db) < 1.0

    def test_stopband_attenuated_20db(self):
        x = _tone(5.0)
        y = bandpass_filter(x, 0.1, 0.7, FS)
        core = slice(200, -200)
        ratio_db = 10 * np.log10(_power(y[core]) / _power(x[core]))
        assert ratio_db < -20.0

    def test_zero_phase(self):
        x = _tone(0.3)
        y = bandpass_filter(x, 0.1, 0.7, FS)
        core = slice(200, -200)
        lags = np.arange(-40, 41)
        xc = [np.dot(x[core], np.roll(y, lag)[core]) for lag in lags]
        assert lags[int(np.argmax(xc))] == 0

    def test_filters_along_time_axis_of_matrix(self):
        x = np.stack([_tone(0.3), _tone(5.0)], axis=1)
        y = bandpass_filter(x, 0.1, 0.7, FS)
        core = slice(200, -200)
        assert _power(y[core, 0]) > 100 * _power(y[core, 1])

    def test_rejects_bad_band(self):
        with pytest.raises(ValueError):
            bandpass_filter(_tone(0.3), 0.7, 0.1, FS)


class TestHampel:
    def test_replaces_all_injected_spikes(self):
        rng = np.random.default_rng(0)
        x = _tone(0.3) + rng.normal(0, 0.05, int(60 * FS))
        spike_idx = rng.choice(x.shape[0], 10, replace=False)
        corrupted = x.copy()
        corrupted[spike_idx] += 15.0
        cleaned = hampel_filter(corrupted)
        assert np.all(np.abs(cleaned[spike_idx] - x[spike_idx]) < 1.0)

    def test_clean_samples_untouched(self):
        x = _tone(0.3)
        corrupted = x.copy()
        corrupted[100] += 15.0
        cleaned = hampel_filter(corrupted)
        untouched = np.delete(np.arange(x.shape[0]), np.arange(95, 106))
        np.testing.assert_allclose(cleaned[untouched], x[untouched])


class TestPcaDenoise:
    def test_improves_snr_on_rank1_signal(self):
        rng = np.random.default_rng(1)
        t = _tone(0.3)
        weights = rng.uniform(0.5, 1.5, 30)
        clean = np.outer(t, weights)
        noisy = clean + rng.normal(0, 0.5, clean.shape)
        reconstructed, components = pca_denoise(noisy, n_components=1)
        err_before = _power(noisy - clean)
        err_after = _power(reconstructed - clean)
        assert err_after < err_before / 5
        assert components.shape == (t.shape[0], 1)
