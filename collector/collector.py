"""CSI collector: serial (or replayed log) -> session .npz + live fanout.

Runs on the laptop during development and on the Raspberry Pi 5 in deployment.

    uv run python -m collector.collector --port /dev/ttyACM0
    uv run python -m collector.collector --replay data/raw/synthetic_15bpm.log

Frames stream into an in-memory buffer that (a) rotates into canonical session .npz
files every --rotate-s seconds and (b) is served to live subscribers (the dashboard's
WebSocket relay) over a localhost TCP socket as newline-delimited JSON:

    {"t": <unix seconds>, "rssi": int, "amp": [float per subcarrier]}

Raw serial lines are also appended to a .log file so any session can be re-parsed
later - raw data is kept, re-collection is expensive.
"""

from __future__ import annotations

import argparse
import json
import socket
import threading
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from csi.io.esp32 import CsiFrame, try_parse_frame
from csi.io.writer import Session, SessionMeta, save_session

FANOUT_HOST = "127.0.0.1"
FANOUT_PORT = 8765


@dataclass
class CollectorConfig:
    port: str | None = None  # serial device, e.g. /dev/ttyACM0
    baud: int = 921600
    replay: Path | None = None  # replay a captured log instead of reading serial
    replay_fs: float = 20.0  # frames per second to emit in replay mode
    out_dir: Path = Path("data/processed/esp32")
    raw_dir: Path = Path("data/raw/esp32")
    rotate_s: float = 600.0  # session file length
    room: str | None = None
    node_positions: str | None = None
    occupants: list[str] = field(default_factory=list)
    label: str | None = None
    fanout_port: int = FANOUT_PORT  # 0 = ephemeral (tests)


class LiveFanout:
    """Localhost TCP publisher: every connected client gets every frame as JSON lines."""

    def __init__(self, host: str = FANOUT_HOST, port: int = FANOUT_PORT) -> None:
        self._clients: list[socket.socket] = []
        self._lock = threading.Lock()
        self._server = socket.create_server((host, port), reuse_port=False)
        self._server.settimeout(1.0)
        self._running = True
        threading.Thread(target=self._accept_loop, daemon=True).start()

    def _accept_loop(self) -> None:
        while self._running:
            try:
                client, _ = self._server.accept()
            except TimeoutError:
                continue
            except OSError:
                return
            with self._lock:
                self._clients.append(client)

    def publish(self, frame: CsiFrame, wall_time: float) -> None:
        payload = (
            json.dumps(
                {
                    "t": wall_time,
                    "rssi": frame.rssi,
                    "amp": frame.amplitude.round(3).tolist(),
                }
            )
            + "\n"
        ).encode()
        with self._lock:
            for client in self._clients[:]:
                try:
                    client.sendall(payload)
                except OSError:
                    self._clients.remove(client)
                    client.close()

    def close(self) -> None:
        self._running = False
        self._server.close()
        with self._lock:
            for client in self._clients:
                client.close()
            self._clients.clear()


class SessionRotator:
    """Accumulates frames and writes a session .npz every rotate_s seconds."""

    def __init__(self, cfg: CollectorConfig) -> None:
        self.cfg = cfg
        self._amps: list[np.ndarray] = []
        self._times: list[float] = []

    def add(self, frame: CsiFrame, wall_time: float) -> None:
        self._amps.append(frame.amplitude)
        self._times.append(wall_time)
        if self._times[-1] - self._times[0] >= self.cfg.rotate_s:
            self.flush()

    def flush(self) -> Path | None:
        if len(self._amps) < 2:
            return None
        # subcarrier count can change if the TX renegotiates; keep the dominant shape
        n_sub = max({a.shape[0] for a in self._amps}, key=[a.shape[0] for a in self._amps].count)
        amps = [a for a in self._amps if a.shape[0] == n_sub]
        times = [t for a, t in zip(self._amps, self._times, strict=True) if a.shape[0] == n_sub]
        amp = np.stack(amps)
        duration = times[-1] - times[0]
        fs = (len(times) - 1) / duration if duration > 0 else 1.0
        started = datetime.fromtimestamp(times[0], tz=UTC)
        meta = SessionMeta(
            dataset="esp32",
            label=self.cfg.label,
            room=self.cfg.room,
            node_positions=self.cfg.node_positions,
            occupants=self.cfg.occupants,
            started_at=started.isoformat(),
        )
        path = self.cfg.out_dir / f"session_{started.strftime('%Y%m%dT%H%M%SZ')}.npz"
        save_session(path, Session(amp=amp, fs=fs, meta=meta))
        print(f"[collector] wrote {path} ({amp.shape[0]} frames @ {fs:.1f} Hz)")
        self._amps.clear()
        self._times.clear()
        return path


def _iter_serial(cfg: CollectorConfig):
    import serial  # local import: pyserial is unused in replay mode/tests

    with serial.Serial(cfg.port, cfg.baud, timeout=5) as dev:
        print(f"[collector] reading {cfg.port} @ {cfg.baud}")
        while True:
            line = dev.readline().decode(errors="replace")
            if line:
                yield line, time.time()


def _iter_replay(cfg: CollectorConfig):
    assert cfg.replay is not None
    period = 1.0 / cfg.replay_fs
    print(f"[collector] replaying {cfg.replay} at {cfg.replay_fs} fps")
    with cfg.replay.open() as fh:
        for line in fh:
            yield line, time.time()
            time.sleep(period)


def run(cfg: CollectorConfig, max_frames: int | None = None) -> None:
    if (cfg.port is None) == (cfg.replay is None):
        raise SystemExit("specify exactly one of --port or --replay")

    cfg.raw_dir.mkdir(parents=True, exist_ok=True)
    raw_path = cfg.raw_dir / f"raw_{datetime.now(tz=UTC).strftime('%Y%m%dT%H%M%SZ')}.log"
    fanout = LiveFanout(port=cfg.fanout_port)
    rotator = SessionRotator(cfg)
    lines = _iter_serial(cfg) if cfg.port else _iter_replay(cfg)
    n = 0
    try:
        with raw_path.open("w") as raw:
            for line, wall_time in lines:
                raw.write(line if line.endswith("\n") else line + "\n")
                frame = try_parse_frame(line)
                if frame is None:
                    continue
                fanout.publish(frame, wall_time)
                rotator.add(frame, wall_time)
                n += 1
                if max_frames is not None and n >= max_frames:
                    break
    except KeyboardInterrupt:
        pass
    finally:
        rotator.flush()
        fanout.close()
        print(f"[collector] done, {n} frames, raw log at {raw_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", help="serial device, e.g. /dev/ttyACM0")
    parser.add_argument("--baud", type=int, default=921600)
    parser.add_argument("--replay", type=Path, help="replay a captured serial log")
    parser.add_argument("--replay-fs", type=float, default=20.0)
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed/esp32"))
    parser.add_argument("--rotate-s", type=float, default=600.0)
    parser.add_argument("--room")
    parser.add_argument("--node-positions", help="rough sketch coords, e.g. 'A:(0,0) B:(3,1)'")
    parser.add_argument("--occupants", nargs="*", default=[], help="who/what is in the room")
    parser.add_argument("--label")
    args = parser.parse_args()
    run(
        CollectorConfig(
            port=args.port,
            baud=args.baud,
            replay=args.replay,
            replay_fs=args.replay_fs,
            out_dir=args.out_dir,
            rotate_s=args.rotate_s,
            room=args.room,
            node_positions=args.node_positions,
            occupants=args.occupants,
            label=args.label,
        )
    )


if __name__ == "__main__":
    main()
