"""Parser for the esp-csi `csi_recv` serial output (ESP32-S3).

Format verified against esp-csi @ examples/get-started/csi_recv/main/app_main.c and
tools/csi_data_read_parse.py. Each frame is one CSV line:

    CSI_DATA,<seq>,<mac>,rssi,rate,sig_mode,mcs,bandwidth,smoothing,not_sounding,
    aggregation,stbc,fec_coding,sgi,noise_floor,ampdu_cnt,channel,secondary_channel,
    local_timestamp,ant,sig_len,rx_format,len,first_word,"[i0,r0,i1,r1,...]"

The trailing quoted array holds interleaved int8 (imag, real) pairs, one pair per
subcarrier: complex[k] = data[2k+1] + 1j * data[2k]. `len` is the byte count, so the
subcarrier count is len // 2. `local_timestamp` is the radio timestamp in microseconds.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

FRAME_PREFIX = "CSI_DATA"
_N_FIELDS = 25  # fields in the ESP32-S3 (non C5/C6) format, including type and data


@dataclass
class CsiFrame:
    seq: int
    mac: str
    rssi: int
    rate: int
    mcs: int
    channel: int
    noise_floor: int
    timestamp_us: int
    sig_len: int
    csi: np.ndarray  # complex64, one entry per subcarrier

    @property
    def amplitude(self) -> np.ndarray:
        return np.abs(self.csi).astype(np.float32)

    @property
    def n_subcarriers(self) -> int:
        return int(self.csi.shape[0])


class FrameParseError(ValueError):
    pass


def parse_frame(line: str) -> CsiFrame:
    """Parse one serial line. Raises FrameParseError on malformed input."""
    start = line.find(FRAME_PREFIX)
    if start < 0:
        raise FrameParseError("no CSI_DATA prefix")
    line = line[start:].strip()

    array_start = line.find('"[')
    array_end = line.rfind(']"')
    if array_start < 0 or array_end <= array_start:
        raise FrameParseError("missing quoted CSI array")

    head = line[:array_start].rstrip(",").split(",")
    if len(head) != _N_FIELDS - 1:
        raise FrameParseError(f"expected {_N_FIELDS - 1} header fields, got {len(head)}")

    try:
        raw = np.array(
            [int(tok) for tok in line[array_start + 2 : array_end].split(",")], dtype=np.int16
        )
    except ValueError as exc:
        raise FrameParseError(f"bad CSI array: {exc}") from exc
    if raw.size < 2 or raw.size % 2 != 0:
        raise FrameParseError(f"CSI array length {raw.size} is not a positive even number")

    try:
        seq = int(head[1])
        mac = head[2]
        rssi = int(head[3])
        rate = int(head[4])
        mcs = int(head[6])
        noise_floor = int(head[14])
        channel = int(head[16])
        timestamp_us = int(head[18])
        sig_len = int(head[20])
        declared_len = int(head[22])
    except (ValueError, IndexError) as exc:
        raise FrameParseError(f"bad header field: {exc}") from exc

    if declared_len != raw.size:
        raise FrameParseError(f"declared len {declared_len} != array size {raw.size}")

    # interleaved (imag, real) int8 pairs -> complex, per csi_data_read_parse.py
    csi = (raw[1::2] + 1j * raw[0::2]).astype(np.complex64)
    return CsiFrame(
        seq=seq,
        mac=mac,
        rssi=rssi,
        rate=rate,
        mcs=mcs,
        channel=channel,
        noise_floor=noise_floor,
        timestamp_us=timestamp_us,
        sig_len=sig_len,
        csi=csi,
    )


def try_parse_frame(line: str) -> CsiFrame | None:
    """Lenient variant: returns None instead of raising. Serial noise happens."""
    if FRAME_PREFIX not in line:
        return None
    try:
        return parse_frame(line)
    except FrameParseError:
        return None
