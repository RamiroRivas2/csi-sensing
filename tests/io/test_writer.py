import numpy as np

from csi.io.writer import Session, SessionMeta, load_session, save_session


def test_roundtrip(tmp_path):
    amp = np.random.default_rng(0).uniform(0, 40, (100, 64)).astype(np.float32)
    meta = SessionMeta(
        dataset="esp32",
        label="lying",
        room="bedroom",
        node_positions="TX:(0,0) RX:(2.5,0)",
        occupants=["ramiro", "dog1"],
        started_at="2026-07-10T04:00:00+00:00",
        extra={"firmware": "csi_recv"},
    )
    path = tmp_path / "session.npz"
    save_session(path, Session(amp=amp, fs=20.0, meta=meta))
    loaded = load_session(path)
    np.testing.assert_array_equal(loaded.amp, amp)
    assert loaded.fs == 20.0
    assert loaded.meta == meta
    assert loaded.duration_s == 5.0
    assert loaded.n_subcarriers == 64


def test_rejects_wrong_shape(tmp_path):
    import pytest

    with pytest.raises(ValueError):
        save_session(
            tmp_path / "bad.npz",
            Session(amp=np.zeros(10, dtype=np.float32), fs=20.0, meta=SessionMeta()),
        )
