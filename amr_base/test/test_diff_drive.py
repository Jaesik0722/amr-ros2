import math

from amr_base.diff_drive import DiffDrive, DiffDriveConfig, int32_delta
import pytest

CFG = DiffDriveConfig(wheel_radius=0.1, wheel_separation=0.3752, gear_ratio=4.33,
                      counts_per_motor_rev=60)


def test_int32_wraparound():
    assert int32_delta(-2**31 + 4, 2**31 - 5) == 9
    assert int32_delta(2**31 - 5, -2**31 + 4) == -9
    assert int32_delta(100, 40) == 60


def test_twist_rpm_roundtrip():
    d = DiffDrive(CFG)
    rpm1, rpm2 = d.twist_to_motor_rpm(0.3, -0.8)
    lin, ang = d.motor_rpm_to_twist(rpm1, rpm2)
    assert lin == pytest.approx(0.3) and ang == pytest.approx(-0.8)


def test_matches_cpp_rpm_conversion():
    # C++: rpm = v * (60 / 2pi) / r * gear, channel 1 negated.
    rpm1, rpm2 = DiffDrive(CFG).twist_to_motor_rpm(0.2, 0.0)
    expected = 0.2 * 60.0 / (2 * math.pi) / 0.1 * 4.33
    assert rpm1 == pytest.approx(-expected) and rpm2 == pytest.approx(expected)


def test_straight_one_wheel_revolution():
    d = DiffDrive(CFG)
    n = CFG.counts_per_wheel_rev
    d.update(0, 0, 0, 0)
    d.update(-round(n * 100), round(n * 100), 0, 0)  # 100 revolutions, ch1 mirrored.
    assert d.x == pytest.approx(100 * 2 * math.pi * 0.1, rel=1e-3)
    assert d.y == pytest.approx(0.0, abs=1e-9)
    assert d.theta == pytest.approx(0.0, abs=1e-9)


def test_spin_in_place():
    d = DiffDrive(CFG)
    d.update(0, 0, 0, 0)
    # Quarter turn: each wheel travels (pi/2) * sep/2.
    counts = (math.pi / 2) * (CFG.wheel_separation / 2) / (2 * math.pi * 0.1) \
        * CFG.counts_per_wheel_rev
    d.update(round(counts), round(counts), 0, 0)  # Both channels positive = left back.
    assert d.theta == pytest.approx(math.pi / 2, abs=0.01)
    assert math.hypot(d.x, d.y) == pytest.approx(0.0, abs=1e-6)
