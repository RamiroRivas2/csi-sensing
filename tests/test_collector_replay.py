"""End-to-end: synthetic serial log -> collector replay -> session npz -> breathing rate.

This is the no-hardware proof of the whole ingest chain.
"""

from collector.collector import CollectorConfig, run
from csi.dsp.breathing import estimate_breathing_rate
from csi.io.writer import load_session
from scripts.make_synthetic_log import FS, make_log


def test_replay_to_session_to_breathing(tmp_path, monkeypatch):
    log = tmp_path / "synthetic.log"
    make_log(log, bpm=15.0, duration_s=90.0)

    cfg = CollectorConfig(
        replay=log,
        replay_fs=2000.0,  # no need to replay in real time inside a test
        out_dir=tmp_path / "processed",
        raw_dir=tmp_path / "raw",
        rotate_s=10_000.0,  # single session, flushed at shutdown
        room="testroom",
        occupants=["synthetic"],
        fanout_port=0,
    )
    run(cfg)

    sessions = list((tmp_path / "processed").glob("*.npz"))
    assert len(sessions) == 1
    session = load_session(sessions[0])
    assert session.n_subcarriers == 64
    assert session.amp.shape[0] > 1000

    # session timing must come from the frames' own radio timestamps, not from
    # the (much faster) replay pacing, so the stored fs is the original rate
    assert abs(session.fs - FS) < 0.5
    est = estimate_breathing_rate(session.amp, session.fs)
    assert abs(est.bpm - 15.0) <= 1.0

    raw_logs = list((tmp_path / "raw").glob("*.log"))
    assert len(raw_logs) == 1 and raw_logs[0].stat().st_size > 0
