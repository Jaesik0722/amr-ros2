"""MD Robot RS485 motor driver protocol (PNT dual-channel commands).

Frame layout: [RMID, TMID, ID, PID, LEN, DATA..., CHECKSUM]
The checksum is the two's complement of the byte sum, so the sum of a valid
frame (checksum included) is 0 modulo 256.
"""

from dataclasses import dataclass
import struct
from typing import Optional

MD_DRIVER_ID = 183  # RMID of the MD driver.
PC_ID = 184  # TMID of the host PC.
DEVICE_ID = 1

PID_DEFAULT_SET = 3
PID_REQ_PID_DATA = 4
PID_TQ_OFF = 5
PID_BRAKE = 6
PID_USE_LIMIT_SW = 17
PID_HALL_TYPE = 21
PID_BAUDRATE = 135
PID_ENC_PPR = 156
PID_MAIN_DATA = 193
PID_PNT_VEL_CMD = 207
PID_PNT_MAIN_DATA = 210

DEFAULT_SETTINGS_BYTE = 0x55
WRITE_CHECK_BYTE = 0xAA

RETURN_NONE = 0
RETURN_PNT_MONITOR = 1
RETURN_PNT_MAIN_DATA = 2

HEADER_LEN = 5
# rpm_l, cur_l, status_l, pos_l, rpm_r, cur_r, status_r, pos_r
_MAIN_DATA_FMT = struct.Struct('<hhBihhBi')
_VEL_CMD_FMT = struct.Struct('<BhBhB')
MAIN_DATA_FRAME_LEN = HEADER_LEN + _MAIN_DATA_FMT.size + 1  # 24 bytes.


def checksum(data: bytes) -> int:
    return (-sum(data)) & 0xFF


def build_frame(pid: int, payload: bytes) -> bytes:
    frame = bytes([MD_DRIVER_ID, PC_ID, DEVICE_ID, pid, len(payload)]) + payload
    return frame + bytes([checksum(frame)])


def torque_off() -> bytes:
    return build_frame(PID_TQ_OFF, b'\x00')


def brake() -> bytes:
    return build_frame(PID_BRAKE, b'\x00')


def reset_default_settings() -> bytes:
    return build_frame(PID_DEFAULT_SET, bytes([DEFAULT_SETTINGS_BYTE]))


def set_baudrate(baudrate_id: int) -> bytes:
    """Set the driver baudrate, baudrate_id 2: 19200, 4: 57600."""
    return build_frame(PID_BAUDRATE, bytes([WRITE_CHECK_BYTE, baudrate_id]))


def use_limit_switch(value: int) -> bytes:
    return build_frame(PID_USE_LIMIT_SW, bytes([value]))


def hall_type(pole_pairs: int) -> bytes:
    """Set the hall sensor type, pole_pairs = number of poles / 2 (MD RS485 manual p.12)."""
    return build_frame(PID_HALL_TYPE, bytes([pole_pairs]))


def encoder_ppr(ppr: int) -> bytes:
    return build_frame(PID_ENC_PPR, struct.pack('<H', ppr))


def request_main_data() -> bytes:
    return build_frame(PID_REQ_PID_DATA, bytes([PID_MAIN_DATA]))


def velocity_command(rpm_ch1: int, rpm_ch2: int,
                     request: int = RETURN_PNT_MAIN_DATA) -> bytes:
    """Command both channels in motor RPM and ask for a reply."""
    rpm_ch1 = max(-32768, min(32767, int(rpm_ch1)))
    rpm_ch2 = max(-32768, min(32767, int(rpm_ch2)))
    return build_frame(PID_PNT_VEL_CMD, _VEL_CMD_FMT.pack(1, rpm_ch1, 1, rpm_ch2, request))


@dataclass
class MainData:
    rpm_ch1: int
    current_ch1: float  # Amperes.
    status_ch1: int
    position_ch1: int
    rpm_ch2: int
    current_ch2: float  # Amperes.
    status_ch2: int
    position_ch2: int


def parse_main_data(frame: bytes) -> MainData:
    (rpm1, cur1, st1, pos1, rpm2, cur2, st2, pos2) = _MAIN_DATA_FMT.unpack_from(frame, HEADER_LEN)
    return MainData(rpm1, cur1 / 10.0, st1, pos1, rpm2, cur2 / 10.0, st2, pos2)


class FrameParser:
    """Re-synchronising stream parser for PNT_MAIN_DATA replies.

    The original C++ library read a fixed 24 bytes and dropped the frame on a
    checksum mismatch, which never recovers once the stream is misaligned.
    This parser scans for a valid header and checksum instead.
    """

    def __init__(self):
        self._buf = bytearray()

    def feed(self, data: bytes) -> Optional[MainData]:
        """Feed raw bytes; return the newest valid frame found, if any."""
        self._buf.extend(data)
        latest = None
        i = 0
        while len(self._buf) - i >= MAIN_DATA_FRAME_LEN:
            if (self._buf[i + 3] == PID_PNT_MAIN_DATA
                    and self._buf[i + 4] == _MAIN_DATA_FMT.size
                    and sum(self._buf[i:i + MAIN_DATA_FRAME_LEN]) & 0xFF == 0):
                latest = parse_main_data(bytes(self._buf[i:i + MAIN_DATA_FRAME_LEN]))
                i += MAIN_DATA_FRAME_LEN
            else:
                i += 1
        del self._buf[:i]
        return latest
