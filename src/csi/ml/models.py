"""Classical classifiers for CSI activity recognition.

Random Forest is the baseline; SVM (RBF) is the comparison. Both are CPU-only
sklearn pipelines - no deep learning, per the project's hardware constraint.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


@dataclass
class RFConfig:
    n_estimators: int = 300
    max_depth: int | None = None
    min_samples_leaf: int = 1
    random_state: int = 42


@dataclass
class SVMConfig:
    C: float = 10.0
    gamma: str = "scale"
    random_state: int = 42


@dataclass
class TrainConfig:
    dataset_root: str = "data/raw/uthar"
    seed: int = 42
    pca_components: int = 6
    rf: RFConfig = field(default_factory=RFConfig)
    svm: SVMConfig = field(default_factory=SVMConfig)


def make_rf(cfg: RFConfig) -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "rf",
                RandomForestClassifier(
                    n_estimators=cfg.n_estimators,
                    max_depth=cfg.max_depth,
                    min_samples_leaf=cfg.min_samples_leaf,
                    random_state=cfg.random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def make_svm(cfg: SVMConfig) -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("svm", SVC(C=cfg.C, gamma=cfg.gamma, random_state=cfg.random_state)),
        ]
    )
