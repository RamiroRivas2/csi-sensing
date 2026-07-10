"""Dashboard backend.

    uv run uvicorn server.app:app --reload --port 8000

REST serves recorded sessions (binary float32 for arrays, JSON for metadata);
/ws/live relays the collector's localhost fanout to browser WebSockets.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import Response

from collector.collector import FANOUT_HOST, FANOUT_PORT
from csi.dsp.breathing import breathing_timeline, estimate_breathing_rate
from csi.dsp.falls import detect_falls
from csi.dsp.features import spectrogram as make_spectrogram
from csi.dsp.filters import bandpass_filter, hampel_filter, pca_denoise
from csi.dsp.quality import link_quality
from csi.dsp.sleep import sleep_metrics
from csi.dsp.vitals import (
    activity_timeline,
    classify_activity,
    estimate_heart_rate,
    heart_rate_timeline,
)
from csi.io.writer import load_session
from server import registry
from server.arrays import MAX_COLS_DEFAULT, binary_response, downsample_time

app = FastAPI(title="csi-sensing")
logger = logging.getLogger(__name__)


def _get(session_id: str):
    try:
        return registry.get_session(session_id)
    except FileNotFoundError:
        raise HTTPException(404, f"unknown session {session_id}") from None


@app.get("/api/sessions")
def sessions() -> list[dict]:
    return [info.__dict__ for info in registry.list_sessions()]


@app.get("/api/sessions/{session_id:path}/csi")
def session_csi(
    session_id: str,
    t0: float = 0.0,
    t1: float | None = None,
    max_cols: int = Query(MAX_COLS_DEFAULT, le=8000),
) -> Response:
    s = _get(session_id)
    i0 = max(0, int(t0 * s.fs))
    i1 = s.amp.shape[0] if t1 is None else min(s.amp.shape[0], int(t1 * s.fs))
    if i1 <= i0:
        raise HTTPException(422, "empty time range")
    return binary_response(downsample_time(s.amp[i0:i1], max_cols))


@app.get("/api/sessions/{session_id:path}/signal")
def session_signal(
    session_id: str,
    subcarrier: int = 0,
    low: float = 0.1,
    high: float = 0.7,
    hampel: bool = True,
) -> Response:
    """Raw and filtered traces for one subcarrier, stacked as shape (2, T)."""
    import numpy as np

    s = _get(session_id)
    if not 0 <= subcarrier < s.n_subcarriers:
        raise HTTPException(422, f"subcarrier out of range 0..{s.n_subcarriers - 1}")
    raw = s.amp[:, subcarrier].astype(np.float64)
    x = hampel_filter(raw) if hampel else raw
    if not 0 < low < high < s.fs / 2:
        raise HTTPException(422, f"need 0 < low < high < {s.fs / 2:.2f}")
    filtered = bandpass_filter(x - x.mean(), low, high, s.fs) + x.mean()
    return binary_response(np.stack([raw, filtered]))


@app.get("/api/sessions/{session_id:path}/psd")
def session_psd(session_id: str, subcarrier: int = 0) -> dict:
    from scipy import signal as sp_signal

    s = _get(session_id)
    if not 0 <= subcarrier < s.n_subcarriers:
        raise HTTPException(422, f"subcarrier out of range 0..{s.n_subcarriers - 1}")
    x = s.amp[:, subcarrier]
    freqs, psd = sp_signal.welch(x - x.mean(), fs=s.fs, nperseg=min(1024, x.shape[0]))
    return {"freqs": freqs.tolist(), "psd": psd.tolist()}


@app.get("/api/sessions/{session_id:path}/spectrogram")
def session_spectrogram(session_id: str, subcarrier: int = 0) -> Response:
    s = _get(session_id)
    if not 0 <= subcarrier < s.n_subcarriers:
        raise HTTPException(422, f"subcarrier out of range 0..{s.n_subcarriers - 1}")
    _, _, sxx = make_spectrogram(s.amp[:, subcarrier], s.fs)
    return binary_response(sxx)


@app.get("/api/sessions/{session_id:path}/breathing")
def session_breathing(session_id: str, window_s: float = 30.0, hop_s: float = 5.0) -> dict:
    s = _get(session_id)
    _, components = pca_denoise(s.amp, n_components=1)
    times, estimates = breathing_timeline(components[:, 0], s.fs, window_s, hop_s)
    return {
        "points": [
            {"t": float(t), "bpm": round(e.bpm, 2), "confidence": round(e.confidence, 3)}
            for t, e in zip(times, estimates, strict=True)
        ]
    }


@app.get("/api/sessions/{session_id:path}/vitals")
def session_vitals(session_id: str) -> dict:
    """Breathing, heart-rate (experimental), and activity timelines plus summary stats."""
    import numpy as np

    s = _get(session_id)
    _, components = pca_denoise(s.amp, n_components=1)
    comp = components[:, 0]

    b_times, b_est = breathing_timeline(comp, s.fs, window_s=30.0, hop_s=5.0)
    h_times, h_est = heart_rate_timeline(comp, s.fs, window_s=20.0, hop_s=5.0)
    a_times, a_est = activity_timeline(comp, s.fs, window_s=10.0, hop_s=5.0)

    def points(times, estimates):
        return [
            {"t": float(t), "bpm": round(e.bpm, 2), "confidence": round(e.confidence, 3)}
            for t, e in zip(times, estimates, strict=True)
        ]

    summary: dict = {"duration_s": round(s.duration_s, 1)}
    if b_est:
        summary["breathing_median_bpm"] = round(float(np.median([e.bpm for e in b_est])), 1)
        summary["breathing_confidence"] = round(float(np.median([e.confidence for e in b_est])), 2)
    if h_est:
        summary["heart_median_bpm"] = round(float(np.median([e.bpm for e in h_est])), 1)
        summary["heart_confidence"] = round(float(np.median([e.confidence for e in h_est])), 2)
    if a_est:
        states = [e.state for e in a_est]
        summary["presence_fraction"] = round(1 - states.count("empty") / len(states), 2)
        summary["motion_fraction"] = round(states.count("moving") / len(states), 2)

    return {
        "breathing": points(b_times, b_est),
        "heart": points(h_times, h_est),
        "activity": [
            {
                "t": float(t),
                "state": e.state,
                "presence": e.presence_score,
                "motion": e.motion_score,
            }
            for t, e in zip(a_times, a_est, strict=True)
        ],
        "summary": summary,
    }


@app.get("/api/sessions/{session_id:path}/falls")
def session_falls(session_id: str) -> dict:
    """Burst-then-stillness fall candidates. Research-grade, not a safety device."""
    s = _get(session_id)
    events = detect_falls(s.amp, s.fs)
    return {
        "events": [
            {
                "t": round(e.t, 1),
                "severity": e.severity,
                "stillness_after": e.stillness_after,
                "confidence": e.confidence,
            }
            for e in events
        ]
    }


@app.get("/api/sessions/{session_id:path}/sleep")
def session_sleep(session_id: str) -> dict:
    """Sleep-quality metrics for one session."""
    s = _get(session_id)
    return sleep_metrics(s.amp, s.fs).__dict__


@app.get("/api/sessions/{session_id:path}/quality")
def session_quality(session_id: str) -> dict:
    """Breathing-band SNR and placement verdict for one session."""
    s = _get(session_id)
    return link_quality(s.amp, s.fs).__dict__


@lru_cache(maxsize=256)
def _night_metrics(session_path: str) -> dict:
    # session files are immutable once written, so caching by resolved path is safe
    s = load_session(Path(session_path))
    return sleep_metrics(s.amp, s.fs).__dict__


MIN_NIGHT_S = 3 * 3600  # anything shorter is a daytime recording, not a night


@app.get("/api/wellbeing")
def wellbeing() -> dict:
    """Sleep metrics across all recorded nights, with baseline-deviation flags.

    Indicators only: sustained deviations in sleep efficiency, restlessness, or
    awakenings correlate with mood in the literature, but nothing here is a
    diagnosis of anything.
    """
    import numpy as np

    nights = []
    for info in registry.list_sessions():
        if info.duration_s < MIN_NIGHT_S:
            continue
        path = registry.session_path(info.id)
        nights.append({"id": info.id, "started_at": info.started_at, **_night_metrics(str(path))})
    nights.sort(key=lambda n: n["started_at"] or n["id"])

    flags: list[dict] = []
    baseline_ready = len(nights) >= 7
    if baseline_ready:
        recent, baseline = nights[-3:], nights[:-3]

        def series(rows: list[dict], key: str) -> list[float]:
            return [r[key] for r in rows if r.get(key) is not None]

        for key, direction, label in (
            ("sleep_efficiency", -1, "sleep efficiency dropping"),
            ("restlessness", +1, "restlessness rising"),
            ("awakenings", +1, "more awakenings"),
        ):
            base_vals, recent_vals = series(baseline, key), series(recent, key)
            if not base_vals or not recent_vals:
                continue
            base_mean = float(np.mean(base_vals))
            recent_mean = float(np.mean(recent_vals))
            spread = float(np.std(base_vals)) or 1e-9
            drift = (recent_mean - base_mean) * direction
            if drift > max(2 * spread, 0.15 * abs(base_mean)):
                flags.append(
                    {
                        "metric": key,
                        "message": label,
                        "baseline": round(base_mean, 2),
                        "recent": round(recent_mean, 2),
                    }
                )

    return {"nights": nights, "baseline_ready": baseline_ready, "flags": flags}


@app.get("/api/sessions/{session_id:path}")
def session_meta(session_id: str) -> dict:
    s = _get(session_id)
    return {
        "id": session_id,
        "fs": s.fs,
        "duration_s": round(s.duration_s, 1),
        "n_subcarriers": s.n_subcarriers,
        "meta": s.meta.__dict__,
    }


EXPERIMENTS_DIR = Path("experiments")


@app.get("/api/experiments")
def experiments() -> list[dict]:
    """List experiments that have produced a metrics.json."""
    out = []
    for metrics_file in sorted(EXPERIMENTS_DIR.glob("*/metrics.json")):
        exp_id = metrics_file.parent.name
        config_file = metrics_file.parent / "config.json"
        out.append(
            {
                "id": exp_id,
                "metrics": json.loads(metrics_file.read_text()),
                "config": json.loads(config_file.read_text()) if config_file.exists() else {},
            }
        )
    return out


LIVE_WINDOW_FRAMES = 1200  # ~60 s at 20 fps kept for rolling bpm estimation


@app.websocket("/ws/live")
async def ws_live(ws: WebSocket) -> None:
    """Relay the collector's localhost fanout to a browser WebSocket.

    Forwards each well-formed frame and augments every ~2 s with rolling vitals
    estimates (breathing, experimental heart rate, activity, link quality) over
    the last LIVE_WINDOW_FRAMES frames, plus deduplicated fall alerts. Garbled
    fanout lines are skipped, a mid-stream subcarrier-count change resets the
    rolling window, and a DSP failure skips that estimate tick; none of these
    drop the connection.
    """
    import numpy as np

    await ws.accept()
    try:
        reader, writer = await asyncio.open_connection(FANOUT_HOST, FANOUT_PORT)
    except OSError:
        await ws.send_json({"type": "error", "message": "collector is not running"})
        await ws.close()
        return

    window: list[list[float]] = []
    times: list[float] = []
    last_estimate = 0.0
    last_fall_wall_t = 0.0
    last_dsp_warn = float("-inf")
    n_sub: int | None = None
    try:
        while True:
            line = await reader.readline()
            if not line:
                await ws.send_json({"type": "error", "message": "collector disconnected"})
                break
            try:
                frame = json.loads(line)
                amp = [float(v) for v in frame["amp"]]
                frame_t = float(frame["t"])
                width = len(amp)
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                continue  # skip a garbled fanout line rather than dropping the connection
            await ws.send_json({"type": "frame", **frame})

            # guard against a subcarrier-count change mid-stream (TX renegotiation):
            # a ragged window would make np.asarray build an object array and crash the
            # DSP. Reset the rolling window to the new width instead.
            if n_sub is not None and width != n_sub:
                window.clear()
                times.clear()
            n_sub = width

            window.append(amp)
            times.append(frame_t)
            if len(window) > LIVE_WINDOW_FRAMES:
                window.pop(0)
                times.pop(0)

            span = times[-1] - times[0] if len(times) > 1 else 0.0
            if span > 20.0 and times[-1] - last_estimate >= 2.0:
                last_estimate = times[-1]
                fs = (len(times) - 1) / span
                # DSP can raise on numerically pathological windows; a bad tick should
                # skip its estimate, not tear down the live connection.
                try:
                    matrix = np.asarray(window, dtype=float)
                    _, components = pca_denoise(matrix, n_components=1)
                    comp = components[:, 0]
                    breathing = estimate_breathing_rate(comp, fs)
                    heart = estimate_heart_rate(comp, fs)
                    activity = classify_activity(comp, fs)
                    quality = link_quality(comp, fs)
                    falls = []
                    fall_cutoff = last_fall_wall_t
                    for event in detect_falls(comp, fs):
                        wall_t = times[0] + event.t
                        if wall_t > fall_cutoff + 10.0:
                            fall_cutoff = wall_t
                            falls.append((wall_t, event))
                except Exception:  # noqa: BLE001 - a bad estimate tick must not kill the WS
                    now = time.monotonic()
                    if now - last_dsp_warn >= 30.0:
                        last_dsp_warn = now
                        logger.warning("live DSP tick failed; skipping estimate", exc_info=True)
                    continue

                await ws.send_json(
                    {
                        "type": "vitals",
                        "breathing_bpm": round(breathing.bpm, 1),
                        "breathing_confidence": round(breathing.confidence, 3),
                        "heart_bpm": round(heart.bpm, 1),
                        "heart_confidence": round(heart.confidence, 3),
                        "state": activity.state,
                        "presence": activity.presence_score,
                        "motion": activity.motion_score,
                        "snr_db": quality.snr_db,
                        "quality_score": quality.score,
                        "quality_verdict": quality.verdict,
                    }
                )
                for wall_t, event in falls:
                    last_fall_wall_t = wall_t  # dedupe across rolling windows
                    await ws.send_json(
                        {
                            "type": "fall_alert",
                            "t": wall_t,
                            "severity": event.severity,
                            "confidence": event.confidence,
                        }
                    )
    except (WebSocketDisconnect, ConnectionResetError):
        pass
    finally:
        with contextlib.suppress(Exception):
            writer.close()
