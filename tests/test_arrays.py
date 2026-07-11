import numpy as np

from server.arrays import downsample_time


def test_no_downsample_when_short():
    arr = np.random.default_rng(0).normal(size=(100, 8)).astype(np.float32)
    np.testing.assert_array_equal(downsample_time(arr, max_cols=200), arr)


def test_preserves_troughs_not_just_peaks():
    # a column that is mostly ~0 with one deep negative dip in the first bin
    t = 4000
    arr = np.zeros((t, 1), dtype=np.float32)
    arr[10, 0] = -9.0  # a trough plain max() would erase
    arr[20, 0] = 5.0  # a peak
    out = downsample_time(arr, max_cols=2000)
    assert out.shape[0] <= 2000
    # both the deep trough and the peak survive somewhere in the output
    assert out.min() <= -8.9
    assert out.max() >= 4.9


def test_bounds_columns():
    arr = np.random.default_rng(1).normal(size=(10000, 4)).astype(np.float32)
    out = downsample_time(arr, max_cols=2000)
    assert out.shape[1] == 4
    assert out.shape[0] <= 2000


def test_no_trailing_samples_dropped():
    # length not divisible by stride: the final partial bin must still be pooled
    arr = np.arange(4001, dtype=np.float32).reshape(-1, 1)
    out = downsample_time(arr, max_cols=2000)
    # the global-max sample (last row, value 4000) must appear
    assert out.max() == 4000.0
