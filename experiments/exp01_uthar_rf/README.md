# exp01 - UT-HAR classical baseline

Reproduces published-ballpark accuracy for classical WiFi CSI activity recognition
on the UT-HAR dataset (Intel 5300 NIC, 7 activities). This is the "academic hardware"
reference point for the ESP32-vs-academic comparison (Paper 1) and the cheap-vs-expensive
comparison against literature numbers (Paper 2).

## Run

```bash
uv run python -c "from csi.io.uthar import download_uthar; download_uthar()"   # ~1 GB, once
uv run python -m csi.ml.train --config experiments/exp01_uthar_rf/config.json
```

Outputs `results.md`, `metrics.json`, and `feature_importances.json` in this folder.

## Method

Per 250x90 sample: PCA across the 90 channels, then the shared DSP feature stack
(dominant frequency, band energies, statistical moments) on the top components.
Random Forest baseline vs SVM (RBF). Identical feature pipeline to the live ESP32
path, so numbers are directly comparable to own-hardware results.
