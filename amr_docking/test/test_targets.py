import math

from amr_docking.targets import compose, target_from_markers

OFFSETS = {10: 0.12, 11: -0.12}


def markers_for(target, offsets=OFFSETS):
    """Ground-truth marker detections for a target at (x, y, yaw)."""
    out = {}
    for i, y in offsets.items():
        x, yy, yaw = compose(target, (0.0, y, 0.0))
        out[i] = (x, yy, math.cos(yaw), math.sin(yaw))
    return out


def close(a, b, tol=1e-9):
    return all(abs(p - q) < tol for p, q in zip(a, b))


def test_compose():
    assert close(compose((1.0, 2.0, math.pi / 2), (1.0, 0.0, 0.0)), (1.0, 3.0, math.pi / 2))


def test_two_markers_recover_target():
    target = (1.5, -0.4, 2.8)
    assert close(target_from_markers(markers_for(target), OFFSETS), target)


def test_two_markers_ignore_bad_normals():
    target = (0.8, 0.3, -2.0)
    m = {i: (x, y, 1.0, 0.0) for i, (x, y, _, _) in markers_for(target).items()}
    assert close(target_from_markers(m, OFFSETS), target)


def test_single_marker_uses_normal():
    target = (2.0, 1.0, math.pi)
    m = markers_for(target)
    del m[11]
    assert close(target_from_markers(m, OFFSETS), target)


def test_unknown_markers():
    assert target_from_markers({3: (1.0, 0.0, -1.0, 0.0)}, OFFSETS) is None


def test_cart_pose_straight_behind():
    from amr_docking.cart_towing import cart_pose_in_base
    x, y, yaw = cart_pose_in_base(0.0, 0.58)
    assert abs(x - (-0.1768 - 0.0918 - 0.58)) < 1e-9 and abs(y) < 1e-9 and yaw == 0.0


def test_convex_hull_square():
    from amr_docking.cart_towing import convex_hull
    hull = convex_hull([(0, 0), (1, 0), (1, 1), (0, 1), (0.5, 0.5)])
    assert sorted(hull) == [(0, 0), (0, 1), (1, 0), (1, 1)]


def simulate_reverse(start_err_y, start_heading, steps=600, dt=0.05):
    """Integrate a unicycle backing into a target at the origin (heading 0)."""
    from amr_docking.reverse_docking import reverse_command
    x, y, th = 0.8, start_err_y, start_heading     # robot in the target frame
    for _ in range(steps):
        c, s = math.cos(th), math.sin(th)
        target = (-(c * x + s * y), -(-s * x + c * y), -th)   # target in the robot frame
        v, w, remaining = reverse_command(target)
        if remaining <= 0.012:
            break
        x += v * math.cos(th) * dt
        y += v * math.sin(th) * dt
        th += w * dt
    return x, y, th


def test_reverse_docking_converges():
    for err_y, heading in ((0.05, 0.0), (-0.08, 0.1), (0.0, -0.15)):
        x, y, th = simulate_reverse(err_y, heading)
        assert x < 0.02 and abs(y) < 0.01 and abs(th) < 0.05, (err_y, heading, x, y, th)
