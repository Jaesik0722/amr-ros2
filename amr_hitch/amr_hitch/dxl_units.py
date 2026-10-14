"""Dynamixel X-series unit conversions (ROS independent)."""

import math

TICKS_PER_REV = 4096
VELOCITY_UNIT_RPM = 0.229

# Control table, Protocol 2.0, X-series.
ADDR_OPERATING_MODE = 11
ADDR_TORQUE_ENABLE = 64
ADDR_GOAL_CURRENT = 102
ADDR_PROFILE_ACCELERATION = 108
ADDR_PROFILE_VELOCITY = 112
ADDR_GOAL_POSITION = 116
ADDR_PRESENT_CURRENT = 126  # 2 bytes, followed by velocity (4) and position (4).
ADDR_PRESENT_VELOCITY = 128
ADDR_PRESENT_POSITION = 132
PRESENT_BLOCK_LEN = 10

OPERATING_MODES = {
    'position': 3,
    'current_based_position': 5,
}


def to_signed(value: int, nbytes: int) -> int:
    bits = 8 * nbytes
    value &= (1 << bits) - 1
    return value - (1 << bits) if value >= 1 << (bits - 1) else value


def ticks_to_rad(ticks: int, zero: int, direction: int) -> float:
    return direction * (ticks - zero) * 2.0 * math.pi / TICKS_PER_REV


def rad_to_ticks(rad: float, zero: int, direction: int, lower: float, upper: float) -> int:
    rad = max(lower, min(upper, rad))
    return zero + round(direction * rad * TICKS_PER_REV / (2.0 * math.pi))


def velocity_to_rad_s(raw: int, direction: int) -> float:
    return direction * raw * VELOCITY_UNIT_RPM * 2.0 * math.pi / 60.0
