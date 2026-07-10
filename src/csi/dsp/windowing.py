"""Sliding-window segmentation of CSI streams."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class WindowConfig:
    window_s: float = 3.0
    overlap: float = 0.5  # fraction of window shared with the next one

    def __post_init__(self) -> None:
        if not 0 <= self.overlap < 1:
            raise ValueError(f"overlap must be in [0, 1), got {self.overlap}")
        if self.window_s <= 0:
            raise ValueError(f"window_s must be positive, got {self.window_s}")


def sliding_windows(x: np.ndarray, fs: float, cfg: WindowConfig) -> np.ndarray:
    """Segment a signal into overlapping windows along axis 0.

    x: (T,) or (T, S). Returns (n_windows, win_len) or (n_windows, win_len, S).
    Trailing samples that do not fill a window are dropped.
    """
    win_len = int(round(cfg.window_s * fs))
    if win_len < 1:
        raise ValueError("window shorter than one sample")
    hop = max(1, int(round(win_len * (1 - cfg.overlap))))
    n = x.shape[0]
    if n < win_len:
        return np.empty((0, win_len, *x.shape[1:]), dtype=x.dtype)
    starts = range(0, n - win_len + 1, hop)
    return np.stack([x[s : s + win_len] for s in starts])


def window_times(n_samples: int, fs: float, cfg: WindowConfig) -> np.ndarray:
    """Center timestamp (seconds) of each window sliding_windows would produce."""
    win_len = int(round(cfg.window_s * fs))
    hop = max(1, int(round(win_len * (1 - cfg.overlap))))
    if n_samples < win_len:
        return np.empty(0)
    starts = np.arange(0, n_samples - win_len + 1, hop)
    return (starts + win_len / 2) / fs
