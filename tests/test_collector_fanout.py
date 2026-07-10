"""LiveFanout must never let one bad subscriber stall the ingest loop."""

import socket
import time

import numpy as np

from collector.collector import LiveFanout
from csi.io.esp32 import CsiFrame


def _big_frame(n_subcarriers: int = 4096) -> CsiFrame:
    csi = np.ones(n_subcarriers, dtype=np.complex64)
    return CsiFrame(
        seq=0,
        mac="aa:bb:cc:dd:ee:ff",
        rssi=-55,
        rate=11,
        mcs=7,
        channel=6,
        noise_floor=-92,
        timestamp_us=0,
        sig_len=60,
        csi=csi,
    )


def _wait_for_clients(fanout: LiveFanout, n: int, timeout_s: float = 2.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        with fanout._lock:
            if len(fanout._clients) >= n:
                return
        time.sleep(0.01)
    raise AssertionError(f"fanout never accepted {n} client(s)")


def test_stalled_subscriber_is_dropped_not_blocking():
    fanout = LiveFanout(port=0)
    port = fanout._server.getsockname()[1]
    stalled = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # tiny receive buffer and never read: the TCP window fills almost immediately
    stalled.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4096)
    stalled.connect(("127.0.0.1", port))
    try:
        _wait_for_clients(fanout, 1)
        frame = _big_frame()
        start = time.monotonic()
        for _ in range(600):  # ~24 MB attempted, far beyond any socket buffer
            fanout.publish(frame, time.time())
            with fanout._lock:
                if not fanout._clients:
                    break
        elapsed = time.monotonic() - start
        with fanout._lock:
            assert fanout._clients == [], "stalled client was never dropped"
        assert elapsed < 5.0, f"publish loop stalled for {elapsed:.1f}s"
    finally:
        stalled.close()
        fanout.close()
