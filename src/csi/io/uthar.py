"""UT-HAR dataset loader.

UT-HAR is a WiFi CSI human activity recognition dataset (Intel 5300 NIC, 7 classes).
We use the processed mirror from the SenseFi benchmark, which ships flattened CSV
splits. Each sample is a 250 x 90 amplitude matrix (250 time steps x 3 antenna
pairs x 30 subcarriers).

Source (verified 2026-07-10): github.com/xyanchen/WiFi-CSI-Sensing-Benchmark links a
Google Drive folder with UT_HAR/data/{X_train,X_val,X_test}.csv and label/y_*.csv.
Fallbacks if the Drive quota blocks: the original ermongroup raw dataset or the
figshare HAR mirror (see download_uthar docstring).

This dataset is the "academic hardware" reference for the ESP32-vs-Intel-5300
comparison. It is only pulled on demand; it is never committed (data/ is gitignored).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

UTHAR_CLASSES: tuple[str, ...] = (
    "lie down",
    "fall",
    "walk",
    "pick up",
    "run",
    "sit down",
    "stand up",
)

N_TIMESTEPS = 250
N_CHANNELS = 90  # 3 antenna pairs x 30 subcarriers

# SenseFi processed mirror. The folder holds four datasets; we only need UT_HAR.zip,
# so we fetch that single file by id rather than the whole (~7 GB) folder.
SENSEFI_DRIVE_FOLDER = "1R0R8SlVbLI1iUFQCzh_mH90H_4CW2iwt"
UTHAR_ZIP_FILE_ID = "1fEiI3nAoOsddR5qcJQXqz4ocM3aMAcwz"
DEFAULT_ROOT = Path("data/raw/uthar")
CACHE_DIR = Path("data/processed/uthar")


def _cache_path(root: Path) -> Path:
    if root.resolve() == DEFAULT_ROOT.resolve():
        return CACHE_DIR / "uthar.npz"
    digest = hashlib.sha1(str(root.resolve()).encode()).hexdigest()[:12]
    return CACHE_DIR / f"uthar-{digest}.npz"


@dataclass
class UTHARData:
    X_train: np.ndarray  # (N, 250, 90) float32
    y_train: np.ndarray  # (N,) int
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray

    @property
    def classes(self) -> tuple[str, ...]:
        return UTHAR_CLASSES


def download_uthar(root: Path = DEFAULT_ROOT) -> None:
    """Download UT_HAR.zip from the SenseFi mirror into ``root`` via gdown.

    Just the UT-HAR file (a few hundred MB), not the whole multi-dataset folder.
    If Google Drive returns a quota error, retry later or fetch manually from one of:
      - github.com/ermongroup/Wifi_Activity_Recognition (raw, ~4 GB)
      - figshare.com/articles/dataset/.../20444538 (direct HTTP, no quota)
    and drop the UT_HAR/{data,label} folders under ``root``.
    """
    import gdown

    root.mkdir(parents=True, exist_ok=True)
    dest = root / "UT_HAR.zip"
    print(f"Downloading UT_HAR.zip from SenseFi Drive into {dest} ...")
    gdown.download(id=UTHAR_ZIP_FILE_ID, output=str(dest), quiet=False)


def _maybe_extract(root: Path) -> None:
    """Extract UT_HAR.zip from the SenseFi mirror if only the zip is present."""
    import zipfile

    zip_path = root / "UT_HAR.zip"
    if zip_path.exists() and not any(root.glob("**/X_train.csv")):
        print(f"extracting {zip_path} ...")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(root)


def _find_split_dir(root: Path) -> Path:
    """Locate the directory holding X_train.csv, tolerating nested UT_HAR/data layouts."""
    _maybe_extract(root)
    for candidate in [root, root / "UT_HAR" / "data", *root.glob("**/")]:
        if (candidate / "X_train.csv").exists():
            return candidate
    raise FileNotFoundError(f"could not find X_train.csv under {root}; run download_uthar() first")


def _load_split(data_dir: Path, label_dir: Path, name: str) -> tuple[np.ndarray, np.ndarray]:
    # SenseFi ships .npy arrays under a .csv extension (NUMPY magic header, not text).
    x = np.load(data_dir / f"X_{name}.csv").astype(np.float32)
    y = np.load(label_dir / f"y_{name}.csv").astype(int)
    x = x.reshape(-1, N_TIMESTEPS, N_CHANNELS)  # already (N, 250, 90), reshape is a no-op guard
    return x, y


def load_uthar(root: Path = DEFAULT_ROOT, use_cache: bool = True) -> UTHARData:
    """Load UT-HAR, caching the parsed arrays to an npz so CSVs parse once."""
    cache = _cache_path(root)
    if use_cache and cache.exists():
        with np.load(cache) as d:
            return UTHARData(
                d["X_train"], d["y_train"], d["X_val"], d["y_val"], d["X_test"], d["y_test"]
            )

    data_dir = _find_split_dir(root)
    label_dir = data_dir.parent / "label"
    if not label_dir.exists():
        label_dir = data_dir  # some mirrors keep labels beside data

    X_train, y_train = _load_split(data_dir, label_dir, "train")
    X_val, y_val = _load_split(data_dir, label_dir, "val")
    X_test, y_test = _load_split(data_dir, label_dir, "test")

    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
    )
    return UTHARData(X_train, y_train, X_val, y_val, X_test, y_test)
