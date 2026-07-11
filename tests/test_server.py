import socket
import threading
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient

from csi.io.writer import Session, SessionMeta, save_session
from server import registry
from server.app import app


@pytest.fixture()
def client(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    t = np.arange(1200) / 20.0
    breathing = np.sin(2 * np.pi * 0.25 * t)  # 15 bpm
    heartbeat = 0.15 * np.sin(2 * np.pi * 1.2 * t)  # 72 bpm, much weaker
    amp = 20 + 2 * np.outer(breathing + heartbeat, rng.uniform(0.5, 1.5, 64))
    amp = (amp + rng.normal(0, 0.3, amp.shape)).astype(np.float32)
    save_session(
        tmp_path / "esp32" / "demo.npz",
        Session(amp=amp, fs=20.0, meta=SessionMeta(dataset="esp32", label="lying")),
    )
    monkeypatch.setattr(registry, "DATA_ROOT", tmp_path)
    registry._load_cached.cache_clear()
    return TestClient(app)


def test_list_sessions(client):
    body = client.get("/api/sessions").json()
    assert [s["id"] for s in body] == ["esp32/demo"]
    assert body[0]["n_subcarriers"] == 64


def test_session_meta(client):
    body = client.get("/api/sessions/esp32/demo").json()
    assert body["fs"] == 20.0
    assert body["meta"]["label"] == "lying"


def test_csi_binary_shape(client):
    res = client.get("/api/sessions/esp32/demo/csi", params={"max_cols": 500})
    assert res.status_code == 200
    shape = tuple(int(v) for v in res.headers["X-Shape"].split(","))
    assert shape[0] <= 500 and shape[1] == 64
    data = np.frombuffer(res.content, dtype=np.float32)
    assert data.shape[0] == shape[0] * shape[1]


def test_signal_raw_and_filtered(client):
    res = client.get("/api/sessions/esp32/demo/signal", params={"subcarrier": 3})
    shape = tuple(int(v) for v in res.headers["X-Shape"].split(","))
    assert shape == (2, 1200)


def test_psd_peak_at_breathing_rate(client):
    body = client.get("/api/sessions/esp32/demo/psd", params={"subcarrier": 3}).json()
    freqs, psd = np.array(body["freqs"]), np.array(body["psd"])
    in_band = (freqs > 0.05) & (freqs < 2.0)
    peak = freqs[in_band][np.argmax(psd[in_band])]
    assert abs(peak - 0.25) < 0.05


def test_breathing_timeline(client):
    body = client.get("/api/sessions/esp32/demo/breathing").json()
    bpms = [p["bpm"] for p in body["points"]]
    assert len(bpms) > 0
    assert abs(np.median(bpms) - 15.0) <= 1.0


def test_vitals(client):
    body = client.get("/api/sessions/esp32/demo/vitals").json()
    assert abs(body["summary"]["breathing_median_bpm"] - 15.0) <= 1.0
    assert abs(body["summary"]["heart_median_bpm"] - 72.0) <= 3.0
    assert body["summary"]["presence_fraction"] > 0.9
    assert body["summary"]["motion_fraction"] < 0.2
    states = {a["state"] for a in body["activity"]}
    assert "still" in states


def test_vitals_heart_path_resists_pure_breathing_forgery(client, tmp_path):
    """A recorded session of pure non-sinusoidal breathing (no cardiac component)
    must not produce a confident heart-rate summary via /vitals."""
    from scipy.signal import sawtooth

    rng = np.random.default_rng(7)
    fs = 20.0
    t = np.arange(int(120 * fs)) / fs
    breathing = sawtooth(2 * np.pi * 0.3 * t, width=0.5)  # triangular 18 bpm
    amp = 20 + 2 * np.outer(breathing, rng.uniform(0.5, 1.5, 64))
    amp = (amp + rng.normal(0, 0.1, amp.shape)).astype(np.float32)
    save_session(
        tmp_path / "esp32" / "breath_only.npz",
        Session(amp=amp, fs=fs, meta=SessionMeta(dataset="esp32", label="breath_only")),
    )
    body = client.get("/api/sessions/esp32/breath_only/vitals").json()
    assert body["summary"]["heart_confidence"] < 0.5


def test_vitals_low_fs_skips_heart_timeline(client, tmp_path):
    """A session recorded below the heart band's nyquist requirement (e.g. a WiFi
    lull) gets an empty heart series, not a 500; breathing and activity remain."""
    rng = np.random.default_rng(3)
    fs = 4.0
    t = np.arange(int(120 * fs)) / fs
    breathing = np.sin(2 * np.pi * 0.25 * t)
    amp = 20 + 2 * np.outer(breathing, rng.uniform(0.5, 1.5, 16))
    amp = (amp + rng.normal(0, 0.3, amp.shape)).astype(np.float32)
    save_session(
        tmp_path / "esp32" / "slow.npz",
        Session(amp=amp, fs=fs, meta=SessionMeta(dataset="esp32")),
    )
    res = client.get("/api/sessions/esp32/slow/vitals")
    assert res.status_code == 200
    body = res.json()
    assert body["heart"] == []
    assert "heart_median_bpm" not in body["summary"]
    assert abs(body["summary"]["breathing_median_bpm"] - 15.0) <= 1.5
    assert len(body["activity"]) > 0


def test_falls_endpoint(client):
    body = client.get("/api/sessions/esp32/demo/falls").json()
    assert body["events"] == []  # quiet breathing session has no falls


def test_sleep_endpoint(client):
    body = client.get("/api/sessions/esp32/demo/sleep").json()
    assert body["sleep_efficiency"] > 0.9
    assert body["awakenings"] == 0


def _save_overnight(tmp_path, name: str, started_at: str, seed: int) -> None:
    rng = np.random.default_rng(seed)
    fs = 2.0
    t = np.arange(int(3.5 * 3600 * fs)) / fs
    breathing = np.sin(2 * np.pi * 0.25 * t)
    amp = 20 + 2 * np.outer(breathing, rng.uniform(0.5, 1.5, 16))
    amp = (amp + rng.normal(0, 0.3, amp.shape)).astype(np.float32)
    save_session(
        tmp_path / "esp32" / f"{name}.npz",
        Session(amp=amp, fs=fs, meta=SessionMeta(dataset="esp32", started_at=started_at)),
    )


def test_wellbeing_endpoint(client, tmp_path):
    body = client.get("/api/wellbeing").json()
    assert body["nights"] == []  # the 60 s demo session is too short to count as a night

    _save_overnight(tmp_path, "night_b", "2026-07-08T22:00:00Z", seed=1)
    _save_overnight(tmp_path, "night_a", "2026-07-09T22:00:00Z", seed=2)
    body = client.get("/api/wellbeing").json()
    # ordered by started_at, not by file path (night_a sorts first lexicographically)
    assert [n["id"] for n in body["nights"]] == ["esp32/night_b", "esp32/night_a"]
    assert body["baseline_ready"] is False
    assert body["flags"] == []


def test_quality_endpoint(client):
    body = client.get("/api/sessions/esp32/demo/quality").json()
    assert body["verdict"] in ("excellent", "good")
    assert body["score"] > 50


def test_unknown_session_404(client):
    assert client.get("/api/sessions/nope/csi").status_code == 404


def _fake_fanout(monkeypatch, payload: bytes) -> None:
    """Serve one connection with a fixed byte payload in place of the collector."""
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    monkeypatch.setattr("server.app.FANOUT_HOST", "127.0.0.1")
    monkeypatch.setattr("server.app.FANOUT_PORT", srv.getsockname()[1])

    def serve() -> None:
        conn, _ = srv.accept()
        conn.sendall(payload)
        conn.close()
        srv.close()

    threading.Thread(target=serve, daemon=True).start()


def test_ws_live_skips_malformed_frames(client, monkeypatch):
    """Garbled fanout lines (bad JSON, missing keys, wrong types) are skipped;
    they must not crash the handler."""
    payload = (
        b"not json\n"
        b'{"amp": [1.0, 2.0]}\n'
        b'{"t": 0.5, "amp": 3}\n'
        b'{"t": 0.7, "amp": [1.0, null]}\n'
        b'{"t": 0.8, "amp": [1.0, "x"]}\n'
        b'{"t": "bad", "amp": [1.0, 2.0]}\n'
        b'{"t": 1.0, "amp": [1.0, 2.0]}\n'
    )
    _fake_fanout(monkeypatch, payload)
    with client.websocket_connect("/ws/live") as ws:
        assert ws.receive_json() == {"type": "frame", "t": 1.0, "amp": [1.0, 2.0]}
        assert ws.receive_json()["type"] == "error"  # collector disconnected


def test_ws_live_ignores_binary_client_frames(client, monkeypatch):
    """A binary websocket frame from the browser is discarded, not treated as a
    protocol error that kills the relay."""
    payload = b'{"t": 1.0, "amp": [1.0, 2.0]}\n'
    _fake_fanout(monkeypatch, payload)
    with client.websocket_connect("/ws/live") as ws:
        ws.send_bytes(b"\x00\x01\x02")
        assert ws.receive_json() == {"type": "frame", "t": 1.0, "amp": [1.0, 2.0]}
        assert ws.receive_json()["type"] == "error"  # collector disconnected


def test_ws_live_dedupes_falls_within_one_tick(client, monkeypatch):
    """Multiple fall events less than 10 s apart in one window emit only the first."""
    # 2 Hz frames: fast enough for the breathing-band frame-rate guard in _live_vitals
    payload = b"".join(f'{{"t": {i * 0.5}, "amp": [1.0, 2.0]}}\n'.encode() for i in range(50))
    rate = SimpleNamespace(bpm=12.0, confidence=0.9)
    monkeypatch.setattr(
        "server.app.pca_denoise", lambda m, n_components=1: (m, np.ones((m.shape[0], 1)))
    )
    monkeypatch.setattr("server.app.estimate_breathing_rate", lambda c, fs: rate)
    monkeypatch.setattr("server.app.estimate_heart_rate", lambda c, fs: rate)
    monkeypatch.setattr(
        "server.app.classify_activity",
        lambda c, fs: SimpleNamespace(state="still", presence_score=1.0, motion_score=0.0),
    )
    monkeypatch.setattr(
        "server.app.link_quality",
        lambda c, fs: SimpleNamespace(snr_db=10.0, score=80, verdict="good"),
    )
    monkeypatch.setattr(
        "server.app.detect_falls",
        lambda c, fs: [
            SimpleNamespace(t=11.0, severity="high", confidence=0.9),
            SimpleNamespace(t=15.0, severity="high", confidence=0.8),
            SimpleNamespace(t=22.0, severity="high", confidence=0.7),
        ],
    )
    _fake_fanout(monkeypatch, payload)
    alerts = []
    with client.websocket_connect("/ws/live") as ws:
        while True:
            msg = ws.receive_json()
            if msg["type"] == "error":
                break
            if msg["type"] == "fall_alert":
                alerts.append(msg["t"])
    assert alerts == [11.0, 22.0]


def test_csi_rejects_nonpositive_max_cols(client):
    for bad in (0, -5):
        res = client.get("/api/sessions/esp32/demo/csi", params={"max_cols": bad})
        assert res.status_code == 422


def test_rewritten_session_is_not_served_stale(client, tmp_path):
    import os

    assert client.get("/api/sessions/esp32/demo").json()["fs"] == 20.0

    path = tmp_path / "esp32" / "demo.npz"
    rng = np.random.default_rng(9)
    amp = (20 + rng.normal(0, 0.3, (600, 64))).astype(np.float32)
    save_session(path, Session(amp=amp, fs=10.0, meta=SessionMeta(dataset="esp32")))
    os.utime(path, ns=(path.stat().st_atime_ns, path.stat().st_mtime_ns + 1_000_000))

    assert client.get("/api/sessions/esp32/demo").json()["fs"] == 10.0


def test_downsample_preserves_fades():
    from server.arrays import downsample_time

    x = np.full((100, 2), 10.0, dtype=np.float32)
    x[10, 0] = 0.0  # deep fade: deviates further from the mean...
    x[11, 0] = 11.0  # ...than this small rise in the same bin
    out = downsample_time(x, max_cols=10)
    assert out.shape == (10, 2)
    assert out[1, 0] < 5.0  # the fade survives; a plain per-bin max would return 11


def test_live_vitals_guards_low_frame_rates():
    from server.app import _live_vitals

    rng = np.random.default_rng(11)

    def matrix(fs: float) -> np.ndarray:
        t = np.arange(int(60 * fs)) / fs
        breathing = np.sin(2 * np.pi * 0.25 * t)
        return (
            20
            + 2 * np.outer(breathing, rng.uniform(0.5, 1.5, 16))
            + rng.normal(0, 0.3, (t.shape[0], 16))
        )

    # below the breathing band's requirement: no estimate, but no exception
    payload, falls = _live_vitals(matrix(1.2), 1.2)
    assert payload is None and falls == []

    # enough for breathing but not for the heart band: heart reads zero
    payload, _ = _live_vitals(matrix(4.0), 4.0)
    assert payload is not None
    assert abs(payload["breathing_bpm"] - 15.0) <= 1.5
    assert payload["heart_bpm"] == 0.0 and payload["heart_confidence"] == 0.0

    # full rate: both estimates present
    payload, _ = _live_vitals(matrix(20.0), 20.0)
    assert payload is not None and payload["heart_bpm"] > 0.0
