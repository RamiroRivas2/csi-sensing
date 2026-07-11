# collector

Ingest service: reads esp-csi `CSI_DATA` lines from the RX board's serial port
(or replays a captured log), then:

- appends every raw line to `data/raw/esp32/raw_*.log` (raw data is never thrown away)
- parses frames and rotates them into canonical session files under `data/processed/esp32/`
- publishes every frame as JSON lines on `127.0.0.1:8765` for live subscribers
  (the FastAPI server's `/ws/live` relay connects here); a subscriber that blocks
  a send for more than 0.2 s is dropped so it can never stall serial ingest

## Usage

```bash
# real hardware
uv run python -m collector.collector --port /dev/ttyACM0 --room bedroom \
    --node-positions "TX:(0,0) RX:(2.5,0)" --occupants ramiro dog1 dog2

# no hardware: replay a captured or synthetic log at real-time speed
uv run python scripts/make_synthetic_log.py           # add --fall-at 60 to inject a fall burst
uv run python -m collector.collector --replay data/raw/synthetic_15bpm.log
```

In replay mode, frame timing is reconstructed from the log's radio timestamps, so
re-parsed sessions keep the original frame rate; `--replay-fs` only paces how fast
lines are played back.

Always pass provenance flags (`--room`, `--node-positions`, `--occupants`, `--label`)
for real recordings - placement studies and the pet-confounder analysis depend on them.

On the Pi 5, install `csi-collector.service` (see the file header) so collection
survives reboots.
