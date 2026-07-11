"""Session discovery and cached loading for the dashboard API."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from csi.io.writer import Session, SessionHeader, load_session, load_session_header

DATA_ROOT = Path("data/processed")


@dataclass
class SessionInfo:
    id: str  # relative path without extension, e.g. "esp32/session_20260710T0400Z"
    dataset: str
    label: str | None
    fs: float
    duration_s: float
    n_subcarriers: int
    started_at: str | None
    room: str | None


def _session_id(path: Path, root: Path) -> str:
    return str(path.relative_to(root).with_suffix(""))


def list_sessions(root: Path | None = None) -> list[SessionInfo]:
    root = root if root is not None else DATA_ROOT
    infos = []
    for path in sorted(root.glob("**/*.npz")):
        try:
            # header only: listing must not decompress every recording's array
            header = _header_cached(str(path.resolve()), path.stat().st_mtime_ns)
        except Exception:
            continue  # unreadable file should not take down the listing
        infos.append(
            SessionInfo(
                id=_session_id(path, root),
                dataset=header.meta.dataset,
                label=header.meta.label,
                fs=header.fs,
                duration_s=round(header.duration_s, 1),
                n_subcarriers=header.n_subcarriers,
                started_at=header.meta.started_at,
                room=header.meta.room,
            )
        )
    return infos


# caches key on (path, mtime) so a rewritten file (e.g. a regenerated demo
# session) is picked up instead of served stale for the life of the process


@lru_cache(maxsize=1024)
def _header_cached(resolved: str, mtime_ns: int) -> SessionHeader:
    return load_session_header(Path(resolved))


@lru_cache(maxsize=8)
def _load_cached(resolved: str, mtime_ns: int) -> Session:
    return load_session(Path(resolved))


def session_path(session_id: str, root: Path | None = None) -> Path:
    root = root if root is not None else DATA_ROOT
    path = (root / session_id).with_suffix(".npz").resolve()
    if root.resolve() not in path.parents and path.parent != root.resolve():
        raise FileNotFoundError(session_id)  # no path traversal
    if not path.exists():
        raise FileNotFoundError(session_id)
    return path


def get_session(session_id: str, root: Path | None = None) -> Session:
    path = session_path(session_id, root)
    return _load_cached(str(path), path.stat().st_mtime_ns)
