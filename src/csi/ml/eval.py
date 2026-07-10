"""Evaluation metrics and results reporting for CSI classifiers."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)


@dataclass
class Metrics:
    accuracy: float
    per_class_f1: dict[str, float]
    macro_f1: float
    confusion: list[list[int]]
    classes: list[str]


def evaluate(y_true: np.ndarray, y_pred: np.ndarray, classes: list[str]) -> Metrics:
    labels = list(range(len(classes)))
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    return Metrics(
        accuracy=round(float(accuracy_score(y_true, y_pred)), 4),
        per_class_f1={classes[i]: round(float(f1[i]), 4) for i in labels},
        macro_f1=round(float(np.mean(f1)), 4),
        confusion=confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        classes=classes,
    )


def write_metrics_json(metrics: dict[str, Metrics], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({k: asdict(v) for k, v in metrics.items()}, indent=2))


def write_results_md(metrics: dict[str, Metrics], path: Path, notes: str = "") -> None:
    classes = next(iter(metrics.values())).classes
    lines = ["# exp01 - UT-HAR classical baseline", ""]
    if notes:
        lines += [notes, ""]

    lines += ["## Accuracy", "", "| model | accuracy | macro F1 |", "| --- | --- | --- |"]
    for name, m in metrics.items():
        lines.append(f"| {name} | {m.accuracy:.3f} | {m.macro_f1:.3f} |")
    lines.append("")

    lines += ["## Per-class F1", "", "| class | " + " | ".join(metrics) + " |"]
    lines.append("| --- | " + " | ".join("---" for _ in metrics) + " |")
    for cls in classes:
        row = " | ".join(f"{m.per_class_f1[cls]:.3f}" for m in metrics.values())
        lines.append(f"| {cls} | {row} |")
    lines.append("")

    best = max(metrics.items(), key=lambda kv: kv[1].accuracy)
    lines += [f"## Confusion matrix ({best[0]})", "", "Rows = true, columns = predicted.", ""]
    header = "| true \\ pred | " + " | ".join(classes) + " |"
    lines += [header, "| --- | " + " | ".join("---" for _ in classes) + " |"]
    for i, cls in enumerate(classes):
        row = " | ".join(str(v) for v in best[1].confusion[i])
        lines.append(f"| {cls} | {row} |")
    lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))
