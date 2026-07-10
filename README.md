# csi-sensing

WiFi CSI (Channel State Information) home sensing. Two ESP32-S3 radios turn a bedroom into a
passive sensor: human motion, down to the chest movement of breathing, disturbs WiFi multipath
propagation, and that disturbance is readable in per-subcarrier amplitude. No cameras, no
microphones, no wearables.

First target: an overnight sleep/breathing monitor. Roadmap: fall detection, presence sensing,
camera-supervised CSI labeling research, and a low-cost 2-node bedside kit.

Everything runs CPU-only: 1D signal processing (scipy) + classical ML (scikit-learn).
No deep learning dependencies.

## Architecture

```
ESP32-S3 TX (csi_send) --WiFi multipath--> ESP32-S3 RX (csi_recv)
       RX --serial--> collector (laptop / Raspberry Pi 5)
       collector --> data/processed/*.npz sessions
       collector --> live fanout --> FastAPI /ws/live
       sessions + dsp --> FastAPI REST (binary float32)
       FastAPI --> React dashboard (live heatmap, DSP inspector, breathing timeline)
```

- `src/csi/io` - ESP32 serial frame parser, canonical session reader/writer
- `src/csi/dsp` - filters, windowing, features, breathing-rate extractor (no ML)
- `src/csi/ml` - Random Forest / SVM baselines
- `src/csi/viz` - matplotlib plots
- `collector/` - ingest service: serial (or replay log) to sessions + live stream
- `server/` - FastAPI backend for the dashboard
- `web/` - React + Vite + TypeScript dashboard
- `firmware/` - ESP32-S3 flashing guide and per-board configs
- `experiments/` - numbered experiments, each with a config and results.md

## Quickstart

```bash
# python side
uv sync
uv run pytest

# end-to-end without hardware: replay a synthetic breathing stream
uv run python scripts/make_synthetic_log.py           # writes data/raw/synthetic_15bpm.log
uv run python -m collector.collector --replay data/raw/synthetic_15bpm.log &
uv run uvicorn server.app:app --reload --port 8000

# dashboard
cd web && npm install && npm run dev                  # http://localhost:5173
```

With hardware, see `firmware/FLASHING.md`, then run the collector against the RX board's
serial port instead of `--replay`.

## Conventions

- Type hints and dataclass configs everywhere; pytest is mandatory for `dsp/` (DSP bugs are silent)
- Amplitude-only for now; ESP32 phase needs CFO/SFO sanitization and is deferred
- Every recording carries provenance metadata: room, node positions, date, occupants (dogs included)
- Every experiment gets a numbered folder with a config file and results.md
- Conventional commits
