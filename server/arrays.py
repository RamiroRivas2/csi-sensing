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
    mean = arr.mean(axis=0, keepdims=True)

    cols = []
    for start in range(0, t, stride):
        chunk = arr[start : start + stride]
        # index of the largest absolute deviation from the column mean, per column
        idx = np.abs(chunk - mean).argmax(axis=0)
        cols.append(np.take_along_axis(chunk, idx[None], axis=0)[0])
    return np.stack(cols)
