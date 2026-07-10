"""Generate a synthetic esp-csi serial log with a breathing signal baked in.

Writes data/raw/synthetic_15bpm.log in the exact CSI_DATA CSV format csi_recv emits,
so the collector's --replay mode exercises the full ingest chain (parser -> sessions
-> live fanout -> dashboard) before any hardware exists.

    uv run python scripts/make_synthetic_log.py --bpm 15 --duration 120
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

N_SUBCARRIERS = 64  # HT20 on ESP32-S3: 128 bytes -> 64 complex values
FS = 20.0  # frames per second


def make_log(
    path: Path, bpm: float, duration_s: float, heart_bpm: float = 72.0, seed: int = 7
) -> None:
    rng = np.random.default_rng(seed)
    n_frames = int(duration_s * FS)
    breathing_hz = bpm / 60.0

    # static multipath baseline per subcarrier + coherent breathing modulation
    # + a much weaker cardiac modulation + noise
    baseline = rng.uniform(8, 40, N_SUBCARRIERS)
    sensitivity = rng.uniform(0.3, 1.0, N_SUBCARRIERS)  # how much each subcarrier "sees"
    t = np.arange(n_frames) / FS
    breathing = np.sin(2 * np.pi * breathing_hz * t)
    heartbeat = np.sin(2 * np.pi * (heart_bpm / 60.0) * t)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for i in range(n_frames):
            amp = baseline * (1 + sensitivity * (0.08 * breathing[i] + 0.012 * heartbeat[i]))
            amp = amp + rng.normal(0, 0.8, N_SUBCARRIERS)
            phase = rng.uniform(0, 2 * np.pi, N_SUBCARRIERS)
            re = np.clip(amp * np.cos(phase), -127, 127).astype(int)
            im = np.clip(amp * np.sin(phase), -127, 127).astype(int)
            interleaved = np.empty(2 * N_SUBCARRIERS, dtype=int)
            interleaved[0::2] = im
            interleaved[1::2] = re
            data = ",".join(str(v) for v in interleaved)
            rssi = -55 + int(rng.integers(-3, 4))
            fh.write(
                f"CSI_DATA,{i},aa:bb:cc:dd:ee:ff,{rssi},11,1,7,1,0,0,0,0,0,0,-92,0,6,0,"
                f'{int(i / FS * 1e6)},0,60,0,{2 * N_SUBCARRIERS},0,"[{data}]"\n'
            )
    print(
        f"wrote {path} ({n_frames} frames, breathing {bpm} bpm, "
        f"heart {heart_bpm} bpm, {N_SUBCARRIERS} subcarriers)"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bpm", type=float, default=15.0)
    parser.add_argument("--heart-bpm", type=float, default=72.0)
    parser.add_argument("--duration", type=float, default=120.0)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    out = args.out or Path(f"data/raw/synthetic_{args.bpm:.0f}bpm.log")
    make_log(out, args.bpm, args.duration, heart_bpm=args.heart_bpm)


if __name__ == "__main__":
    main()
