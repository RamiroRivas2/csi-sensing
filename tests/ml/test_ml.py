"""ML pipeline tests on a tiny synthetic dataset - no UT-HAR download required.

Proves the feature extraction + classifier wiring separates classes with distinct
dominant frequencies, so a green suite means the exp01 pipeline is sound even
before the ~1 GB dataset lands.
"""

import numpy as np

from csi.ml.eval import evaluate
from csi.ml.models import RFConfig, SVMConfig, make_rf, make_svm
from csi.ml.train import build_features

FS = 62.5


def _synthetic_dataset(n_per_class: int = 30, seed: int = 0):
    """3 classes with distinct spatial (per-channel) signatures AND dominant frequencies.

    Mirrors real activities, which differ in both which subcarriers move and how fast -
    so both the per-channel-statistic features and the spectral features are informative.
    """
    rng = np.random.default_rng(seed)
    freqs = [2.0, 6.0, 12.0]
    class_weights = [rng.uniform(0.2, 1.8, 90) for _ in freqs]  # fixed spatial pattern per class
    X, y = [], []
    t = np.arange(250) / FS
    for label, (f, base_w) in enumerate(zip(freqs, class_weights, strict=True)):
        for _ in range(n_per_class):
            signal = np.sin(2 * np.pi * f * t + rng.uniform(0, 2 * np.pi))
            weights = base_w * rng.uniform(0.9, 1.1, 90)  # small per-sample jitter
            sample = np.outer(signal, weights) + rng.normal(0, 0.3, (250, 90))
            X.append(sample.astype(np.float32))
            y.append(label)
    return np.stack(X), np.array(y)


def test_rf_separates_synthetic_classes():
    X, y = _synthetic_dataset()
    F = build_features(X, n_components=3, fs=FS, desc="test")
    split = int(len(y) * 0.7)
    idx = np.random.default_rng(1).permutation(len(y))
    tr, te = idx[:split], idx[split:]
    model = make_rf(RFConfig(n_estimators=100))
    model.fit(F[tr], y[tr])
    m = evaluate(y[te], model.predict(F[te]), ["a", "b", "c"])
    assert m.accuracy > 0.9


def test_svm_also_works():
    X, y = _synthetic_dataset(n_per_class=20, seed=2)
    F = build_features(X, n_components=3, fs=FS, desc="test")
    split = int(len(y) * 0.7)
    idx = np.random.default_rng(3).permutation(len(y))
    tr, te = idx[:split], idx[split:]
    model = make_svm(SVMConfig())
    model.fit(F[tr], y[tr])
    m = evaluate(y[te], model.predict(F[te]), ["a", "b", "c"])
    assert m.accuracy > 0.85


def test_evaluate_shapes():
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 2, 0, 2, 2])
    m = evaluate(y_true, y_pred, ["a", "b", "c"])
    assert 0 <= m.accuracy <= 1
    assert set(m.per_class_f1) == {"a", "b", "c"}
    assert len(m.confusion) == 3 and len(m.confusion[0]) == 3
