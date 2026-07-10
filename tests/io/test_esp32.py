import numpy as np

from csi.io.esp32 import FrameParseError, parse_frame, try_parse_frame

VALID = (
    'CSI_DATA,42,aa:bb:cc:dd:ee:ff,-55,11,1,7,1,0,0,0,0,0,0,-92,0,6,0,1234567,0,60,0,8,0,'
    '"[1,2,3,4,5,6,7,8]"'
)


def test_parses_valid_frame():
    frame = parse_frame(VALID)
    assert frame.seq == 42
    assert frame.mac == "aa:bb:cc:dd:ee:ff"
    assert frame.rssi == -55
    assert frame.channel == 6
    assert frame.timestamp_us == 1234567
    assert frame.n_subcarriers == 4
    # interleaved (imag, real): complex[k] = data[2k+1] + 1j*data[2k]
    np.testing.assert_array_equal(frame.csi, np.array([2 + 1j, 4 + 3j, 6 + 5j, 8 + 7j]))


def test_amplitude():
    frame = parse_frame(VALID)
    np.testing.assert_allclose(frame.amplitude[0], np.hypot(2, 1), rtol=1e-6)


def test_leading_garbage_tolerated():
    assert try_parse_frame("\x00\xffboot noise" + VALID) is not None


def test_truncated_line_rejected():
    assert try_parse_frame(VALID[:50]) is None


def test_corrupted_array_rejected():
    assert try_parse_frame(VALID.replace("5,6", "5,x")) is None


def test_len_mismatch_rejected():
    bad = VALID.replace(",8,0,\"[", ",6,0,\"[")
    assert try_parse_frame(bad) is None


def test_non_csi_line_is_none():
    assert try_parse_frame("I (1234) wifi: connected") is None


def test_strict_raises():
    import pytest

    with pytest.raises(FrameParseError):
        parse_frame("garbage")
