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
    """Strided pooling along axis 0 so payloads stay bounded regardless of length.

    Uses per-bin max of |x - mean| added back around the mean, which preserves the
    visual texture of motion bursts better than plain decimation.
    """
    t = arr.shape[0]
    if t <= max_cols:
        return arr
    stride = int(np.ceil(t / max_cols))
    usable = (t // stride) * stride
    trimmed = arr[:usable]
    binned = trimmed.reshape(-1, stride, *arr.shape[1:])
    return binned.max(axis=1)
