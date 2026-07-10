import numpy as np
import pytest

from csi.dsp.windowing import WindowConfig, sliding_windows, window_times

FS = 20.0


def test_window_count_and_shape():
    x = np.arange(200.0)
    cfg = WindowConfig(window_s=2.0, overlap=0.5)  # win 40, hop 20
    w = sliding_windows(x, FS, cfg)
    assert w.shape == (9, 40)


def test_overlap_alignment():
    x = np.arange(200.0)
    cfg = WindowConfig(window_s=2.0, overlap=0.5)
    w = sliding_windows(x, FS, cfg)
    np.testing.assert_array_equal(w[0, 20:], w[1, :20])


def test_multichannel_shape():
    x = np.zeros((200, 30))
    w = sliding_windows(x, FS, WindowConfig(window_s=2.0, overlap=0.5))
    assert w.shape == (9, 40, 30)


def test_times_match_window_count():
    x = np.zeros(200)
    cfg = WindowConfig(window_s=2.0, overlap=0.5)
    assert window_times(x.shape[0], FS, cfg).shape[0] == sliding_windows(x, FS, cfg).shape[0]


def test_too_short_input():
    w = sliding_windows(np.zeros(10), FS, WindowConfig(window_s=2.0))
    assert w.shape[0] == 0


def test_invalid_overlap_rejected():
    with pytest.raises(ValueError):
        WindowConfig(overlap=1.0)
