"""Differential drive kinematics and encoder odometry (ROS independent)."""

from dataclasses import dataclass
import math


def int32_delta(new: int, old: int) -> int:
    """Difference of two int32 encoder readings, robust to wrap-around."""
    return ((new - old + 2**31) % 2**32) - 2**31


@dataclass
class DiffDriveConfig:
    wheel_radius: float
    wheel_separation: float
    gear_ratio: float  # Motor revolutions per wheel revolution.
    counts_per_motor_rev: float  # Hall: number_of_poles * 3.
    left_sign: int = -1  # Channel 1 (left) is mounted mirrored.
    right_sign: int = 1

    @property
    def counts_per_wheel_rev(self) -> float:
        return self.counts_per_motor_rev * self.gear_ratio


class DiffDrive:

    def __init__(self, cfg: DiffDriveConfig):
        self.cfg = cfg
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        self.linear = 0.0
        self.angular = 0.0
        self.left_angle = 0.0  # Wheel joint angles [rad].
        self.right_angle = 0.0
        self._last_counts = None

    def twist_to_motor_rpm(self, linear: float, angular: float) -> tuple:
        """Return (ch1, ch2) motor RPM commands for a body twist."""
        c = self.cfg
        v_left = linear - angular * c.wheel_separation / 2.0
        v_right = linear + angular * c.wheel_separation / 2.0
        to_rpm = 60.0 / (2.0 * math.pi * c.wheel_radius) * c.gear_ratio
        return c.left_sign * v_left * to_rpm, c.right_sign * v_right * to_rpm

    def motor_rpm_to_twist(self, rpm_ch1: int, rpm_ch2: int) -> tuple:
        c = self.cfg
        to_vel = 2.0 * math.pi * c.wheel_radius / 60.0 / c.gear_ratio
        v_left = c.left_sign * rpm_ch1 * to_vel
        v_right = c.right_sign * rpm_ch2 * to_vel
        return (v_right + v_left) / 2.0, (v_right - v_left) / c.wheel_separation

    def reset(self):
        self.x = self.y = self.theta = 0.0
        self._last_counts = None

    def update(self, counts_ch1: int, counts_ch2: int, rpm_ch1: int, rpm_ch2: int):
        """Integrate pose from encoder counts; velocity from reported RPM.

        Encoder resolution is coarse (~2.4 mm/count with the MDH300), so a
        finite difference over 50 ms is too noisy for velocity.
        """
        self.linear, self.angular = self.motor_rpm_to_twist(rpm_ch1, rpm_ch2)
        if self._last_counts is None:
            self._last_counts = (counts_ch1, counts_ch2)
            return
        c = self.cfg
        d1 = int32_delta(counts_ch1, self._last_counts[0])
        d2 = int32_delta(counts_ch2, self._last_counts[1])
        self._last_counts = (counts_ch1, counts_ch2)

        rad_per_count = 2.0 * math.pi / c.counts_per_wheel_rev
        d_left_angle = c.left_sign * d1 * rad_per_count
        d_right_angle = c.right_sign * d2 * rad_per_count
        self.left_angle += d_left_angle
        self.right_angle += d_right_angle

        d_left = d_left_angle * c.wheel_radius
        d_right = d_right_angle * c.wheel_radius
        d_center = (d_left + d_right) / 2.0
        d_theta = (d_right - d_left) / c.wheel_separation

        # Midpoint integration.
        self.x += d_center * math.cos(self.theta + d_theta / 2.0)
        self.y += d_center * math.sin(self.theta + d_theta / 2.0)
        self.theta = math.atan2(math.sin(self.theta + d_theta), math.cos(self.theta + d_theta))
