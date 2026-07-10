# exp01 - UT-HAR classical baseline

Reproduces published-ballpark accuracy for classical WiFi CSI activity recognition
on the UT-HAR dataset (Intel 5300 NIC, 7 activities). This is the "academic hardware"
reference point for the ESP32-vs-academic comparison (Paper 1) and the cheap-vs-expensive
comparison against literature numbers (Paper 2).

## Run

```bash
uv run python -c "from csi.io.uthar import download_uthar; download_uthar()"   # a few hundred MB, once
uv run python -m csi.ml.train --config experiments/exp01_uthar_rf/config.json
```

Outputs `results.md`, `metrics.json`, `feature_importances.json`, and
`confusion_matrix.png` in this folder.

## Method

Per 250x90 sample: per-channel temporal statistics (mean/std/min/max/ptp/median for
each of the 90 channels) concatenated with the shared DSP feature stack (dominant
frequency, band energies, statistical moments) on the top PCA components. The
per-channel stats carry most of the discriminative power - spectral features on PCA
components alone capped accuracy at ~63%. Random Forest baseline vs SVM (RBF). The
spectral half is the exact feature stack used on the live ESP32 path, so numbers are
comparable to own-hardware results.
