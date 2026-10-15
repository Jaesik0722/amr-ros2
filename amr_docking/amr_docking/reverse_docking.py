"""Reverse docking control law for backing the hitch under a cart's tow bar.

opennav_docking (Jazzy) mishandles backward docking (steering or target projection end
up mirrored), so carts are docked with this small controller on top of the ArUco
detections. The robot drives backwards along the target's x axis.
"""

import math


def clamp(value, low, high):
    return max(low, min(high, value))


def relative(target):
    """Robot pose in the target frame, given the target (dock) pose in the robot frame."""
    tx, ty, tyaw = target
    c, s = math.cos(tyaw), math.sin(tyaw)
    return (-(c * tx + s * ty), -(-s * tx + c * ty), -tyaw)


def reverse_command(target, v_max=0.10, v_min=0.03, k_v=0.5, k_lat=6.0, k_head=1.5,
                    w_max=0.4):
    """Return (v, w, remaining) to back into `target` (x, y, yaw in the robot frame).

    The robot is ahead of the target (remaining > 0) and reverses onto it; lateral and
    heading errors are steered out on the way. remaining <= 0 means arrived.
    """
    xr, yr, th = relative(target)
    th = math.atan2(math.sin(th), math.cos(th))
    v = -clamp(k_v * xr, v_min, v_max)
    w = clamp(k_lat * yr - k_head * th, -w_max, w_max)
    return v, w, xr
