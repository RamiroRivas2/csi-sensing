"""Matplotlib plots for CSI data and experiment results.

Static PNGs for notebooks, papers, and experiment results.md files. The dashboard
renders its own interactive charts; this module is for publication figures.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: no display needed on the Pi or in CI
import matplotlib.pyplot as plt
import numpy as np


def plot_csi_heatmap(amp: np.ndarray, fs: float, path: Path, title: str = "CSI amplitude") -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    extent = (0, amp.shape[0] / fs, 0, amp.shape[1])
    ax.imshow(amp.T, aspect="auto", origin="lower", cmap="viridis", extent=extent)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("subcarrier")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_psd(freqs: np.ndarray, psd: np.ndarray, path: Path, mark_hz: float | None = None) -> None:
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.semilogy(freqs, psd, color="#1f9e89")
    if mark_hz is not None:
        ax.axvline(mark_hz, color="#f472b6", ls="--", label=f"{mark_hz:.2f} Hz")
        ax.legend()
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("PSD")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_breathing_timeline(times: np.ndarray, bpm: np.ndarray, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 3))
    ax.plot(times, bpm, color="#38bdf8")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("breaths / min")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_confusion_matrix(
    confusion: np.ndarray, classes: list[str], path: Path, title: str = "Confusion matrix"
) -> None:
    cm = np.asarray(confusion, dtype=float)
    norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(classes)), classes, rotation=45, ha="right")
    ax.set_yticks(range(len(classes)), classes)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(
                j, i, f"{int(cm[i, j])}", ha="center", va="center",
                color="white" if norm[i, j] > 0.5 else "black", fontsize=8,
            )
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
