"""Binary numpy transport: float32 bytes + shape headers, no JSON overhead."""

from __future__ import annotations

import numpy as np
from fastapi import Response

MAX_COLS_DEFAULT = 2000


def binary_response(arr: np.ndarray) -> Response:
    arr32 = np.ascontiguousarray(arr, dtype=np.float32)
    return Response(
        content=arr32.tobytes(),
        media_type="application/octet-stream",
        headers={
            "X-Shape": ",".join(str(d) for d in arr32.shape),
            "X-Dtype": "float32",
            "Access-Control-Expose-Headers": "X-Shape, X-Dtype",
        },
    )


def downsample_time(arr: np.ndarray, max_cols: int = MAX_COLS_DEFAULT) -> np.ndarray:
    """Pool along axis 0 so payloads stay bounded, keeping the biggest excursion per bin.

    For each bin we keep the sample that deviates most (in absolute value) from that
    column's overall mean, preserving sign. This retains both peaks and troughs of a
    motion burst, unlike plain max pooling (which is upward-biased and erases dips)
    or plain decimation (which drops excursions between kept samples). The last
    partial bin is pooled too so no trailing samples are silently dropped.
    """
    t = arr.shape[0]
    if t <= max_cols:
        return arr
    stride = int(np.ceil(t / max_cols))
    bins = -(-t // stride)
    mean = arr.mean(axis=0, keepdims=True)

    pad = bins * stride - t
    if pad:
        # pad the partial last bin with the column mean: zero deviation, so a
        # pad sample can never win the argmax over a real one
        arr = np.concatenate([arr, np.broadcast_to(mean, (pad, *arr.shape[1:]))])
    chunks = arr.reshape(bins, stride, *arr.shape[1:])
    # index of the largest absolute deviation from the column mean, per bin and column
    idx = np.abs(chunks - mean).argmax(axis=1)
    return np.take_along_axis(chunks, np.expand_dims(idx, axis=1), axis=1).squeeze(axis=1)
