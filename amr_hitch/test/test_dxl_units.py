import math

from amr_hitch import dxl_units as u
import pytest


def test_to_signed():
    assert u.to_signed(0xFFFF, 2) == -1
    assert u.to_signed(0x7FFF, 2) == 32767
    assert u.to_signed(0xFFFFFFFE, 4) == -2


def test_tick_roundtrip_and_direction():
    lim = (-math.pi / 2, math.pi / 2)
    for direction in (1, -1):
        ticks = u.rad_to_ticks(0.5, 2048, direction, *lim)
        assert u.ticks_to_rad(ticks, 2048, direction) == pytest.approx(0.5, abs=2e-3)


def test_command_is_clamped_to_limits():
    assert u.rad_to_ticks(3.0, 2048, 1, -math.pi / 2, math.pi / 2) == 2048 + 1024


def test_velocity_units():
    assert u.velocity_to_rad_s(100, 1) == pytest.approx(100 * 0.229 * 2 * math.pi / 60)
