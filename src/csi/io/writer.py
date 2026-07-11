"""Canonical CSI session storage.

Every data source (ESP32 stream, replayed logs, synthetic signals, public datasets)
normalizes into one on-disk format so the DSP pipeline and the dashboard never care
where data came from:

    session.npz
      amp:  float32 array of shape (T, S) - amplitude per time step and subcarrier
      fs:   float scalar - sample rate in Hz
      meta: JSON string - provenance (room, node positions, occupants, label, ...)
"""

from __future__ import annotations

import json
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class SessionMeta:
    """Provenance for a recording. Dogs count as occupants."""

    dataset: str = "esp32"
    label: str | None = None
    room: str | None = None
    node_positions: str | None = None
    occupants: list[str] = field(default_factory=list)
    started_at: str | None = None  # ISO 8601
    notes: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(self.__dict__)

    @classmethod
    def from_json(cls, payload: str) -> SessionMeta:
        data = json.loads(payload)
        known = {k: data.pop(k) for k in list(data) if k in cls.__dataclass_fields__}
        extra = known.pop("extra", {})
        return cls(**known, extra=extra | data)


@dataclass
class Session:
    amp: np.ndarray  # (T, S) float32
    fs: float
    meta: SessionMeta

    @property
    def duration_s(self) -> float:
        return self.amp.shape[0] / self.fs

    @property
    def n_subcarriers(self) -> int:
        return self.amp.shape[1]


def save_session(path: Path, session: Session) -> None:
    if session.amp.ndim != 2:
        raise ValueError(f"amp must be (T, S), got shape {session.amp.shape}")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        amp=session.amp.astype(np.float32),
        fs=np.float64(session.fs),
        meta=session.meta.to_json(),
    )


def load_session(path: Path) -> Session:
    with np.load(path, allow_pickle=False) as data:
        return Session(
            amp=data["amp"],
            fs=float(data["fs"]),
            meta=SessionMeta.from_json(str(data["meta"])),
        )


@dataclass
class SessionHeader:
    """Everything about a session except the amp array itself."""

    shape: tuple[int, int]  # (T, S)
    fs: float
    meta: SessionMeta

    @property
    def duration_s(self) -> float:
        return self.shape[0] / self.fs

    @property
    def n_subcarriers(self) -> int:
        return self.shape[1]


def load_session_header(path: Path) -> SessionHeader:
    """Read shape, fs, and meta without decompressing the amp array.

    Listing an archive of long sessions must not pay the cost of inflating
    every recording; the amp member's npy header alone carries its shape.
    """
    with zipfile.ZipFile(path) as zf, zf.open("amp.npy") as fh:
        version = np.lib.format.read_magic(fh)
        if version == (1, 0):
            shape, _, _ = np.lib.format.read_array_header_1_0(fh)
        elif version == (2, 0):
            shape, _, _ = np.lib.format.read_array_header_2_0(fh)
        else:
            raise ValueError(f"unsupported npy format version {version} in {path}")
    if len(shape) != 2:
        raise ValueError(f"amp must be (T, S), got shape {shape}")
    with np.load(path, allow_pickle=False) as data:  # fs and meta are tiny members
        return SessionHeader(
            shape=(int(shape[0]), int(shape[1])),
            fs=float(data["fs"]),
            meta=SessionMeta.from_json(str(data["meta"])),
        )
