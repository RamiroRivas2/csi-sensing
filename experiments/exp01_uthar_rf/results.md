# exp01 - UT-HAR classical baseline

Dataset: UT-HAR (lie down, fall, walk, pick up, run, sit down, stand up). Features per sample (606): 540 per-channel temporal statistics (mean/std/min/max/ptp/median over 90 channels) + 66 spectral features on 6 PCA components. Train n=4473 (train+val), test n=500.

Reference: the SenseFi benchmark reports 87.75% for Random Forest on UT-HAR. Our richer feature set lands higher; note also that UT-HAR is built from overlapping sliding windows, so adjacent train/test windows are correlated and all reported UT-HAR numbers (ours and published) are optimistic relative to a fully subject-disjoint split. This is the academic-hardware reference point; the honest comparison is against our own ESP32 data under the identical feature pipeline.

Top features (RF importance): pca0_band_5.0_10.0 (0.017), pca1_band_5.0_10.0 (0.015), pca2_band_5.0_10.0 (0.012), pca3_band_5.0_10.0 (0.011), ch57_mean (0.010), pca4_band_5.0_10.0 (0.007), ch57_median (0.007), pca0_band_2.0_5.0 (0.006).

## Accuracy

| model | accuracy | macro F1 |
| --- | --- | --- |
| random_forest | 0.962 | 0.944 |
| svm_rbf | 0.972 | 0.954 |

## Per-class F1

| class | random_forest | svm_rbf |
| --- | --- | --- |
| lie down | 0.953 | 0.953 |
| fall | 0.955 | 0.967 |
| walk | 0.993 | 1.000 |
| pick up | 0.951 | 0.962 |
| run | 0.980 | 1.000 |
| sit down | 0.868 | 0.875 |
| stand up | 0.909 | 0.918 |

## Confusion matrix (svm_rbf)

Rows = true, columns = predicted.

| true \ pred | lie down | fall | walk | pick up | run | sit down | stand up |
| --- | --- | --- | --- | --- | --- | --- | --- |
| lie down | 61 | 0 | 0 | 3 | 0 | 2 | 0 |
| fall | 0 | 44 | 0 | 1 | 0 | 0 | 0 |
| walk | 0 | 0 | 147 | 0 | 0 | 0 | 0 |
| pick up | 0 | 0 | 0 | 50 | 0 | 0 | 0 |
| run | 0 | 0 | 0 | 0 | 121 | 0 | 0 |
| sit down | 1 | 2 | 0 | 0 | 0 | 35 | 2 |
| stand up | 0 | 0 | 0 | 0 | 0 | 3 | 28 |
