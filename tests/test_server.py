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
