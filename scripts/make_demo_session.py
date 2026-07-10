"""Create a demo session npz by parsing a synthetic log through the real ingest path.

Gives the dashboard something to browse before any hardware recording exists:

    uv run python scripts/make_synthetic_log.py
    uv run python scripts/make_demo_session.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from csi.io.esp32 import try_parse_frame
from csi.io.writer import Session, SessionMeta, save_session

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_synthetic_log import FS  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, default=Path("data/raw/synthetic_15bpm.log"))
    parser.add_argument(
        "--out", type=Path, default=Path("data/processed/demo/synthetic_breathing_15bpm.npz")
    )
    args = parser.parse_args()

    amps = []
    with args.log.open() as fh:
        for line in fh:
            frame = try_parse_frame(line)
            if frame is not None:
                amps.append(frame.amplitude)
    if len(amps) < 2:
        raise SystemExit(f"no frames parsed from {args.log} - run make_synthetic_log.py first")

    session = Session(
        amp=np.stack(amps),
        fs=FS,
        meta=SessionMeta(
            dataset="demo",
            label="synthetic breathing 15 bpm",
            room="none (synthetic)",
            occupants=["synthetic"],
        ),
    )
    save_session(args.out, session)
    print(f"wrote {args.out} ({session.amp.shape[0]} frames @ {FS} Hz)")


if __name__ == "__main__":
    main()
