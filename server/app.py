"""Dashboard backend.

    uv run uvicorn server.app:app --reload --port 8000

REST serves recorded sessions (binary float32 for arrays, JSON for metadata);
/ws/live relays the collector's localhost fanout to browser WebSockets.
"""

from __future__ import annotations

import asyncio
import contextlib
import json

from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import Response

from collector.collector import FANOUT_HOST, FANOUT_PORT
from csi.dsp.breathing import breathing_timeline, estimate_breathing_rate
from csi.dsp.features import spectrogram as make_spectrogram
from csi.dsp.filters import bandpass_filter, hampel_filter, pca_denoise
from server import registry
from server.arrays import MAX_COLS_DEFAULT, binary_response, downsample_time

app = FastAPI(title="csi-sensing")


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


@app.get("/api/experiments")
def experiments() -> list[dict]:
    """Populated once the UT-HAR baseline (exp01) lands."""
    return []


LIVE_WINDOW_FRAMES = 1200  # ~60 s at 20 fps kept for rolling bpm estimation


@app.websocket("/ws/live")
async def ws_live(ws: WebSocket) -> None:
    """Relay the collector's localhost fanout to a browser WebSocket.

    Forwards each frame verbatim and augments every ~2 s with a rolling breathing
    estimate over the last LIVE_WINDOW_FRAMES frames.
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
    try:
        while True:
            line = await reader.readline()
            if not line:
                await ws.send_json({"type": "error", "message": "collector disconnected"})
                break
            frame = json.loads(line)
            await ws.send_json({"type": "frame", **frame})

            window.append(frame["amp"])
            times.append(frame["t"])
            if len(window) > LIVE_WINDOW_FRAMES:
                window.pop(0)
                times.pop(0)

            span = times[-1] - times[0] if len(times) > 1 else 0.0
            if span > 20.0 and times[-1] - last_estimate >= 2.0:
                last_estimate = times[-1]
                fs = (len(times) - 1) / span
                est = estimate_breathing_rate(np.asarray(window), fs)
                await ws.send_json(
                    {
                        "type": "breathing",
                        "bpm": round(est.bpm, 1),
                        "confidence": round(est.confidence, 3),
                    }
                )
    except (WebSocketDisconnect, ConnectionResetError):
        pass
    finally:
        with contextlib.suppress(Exception):
            writer.close()
