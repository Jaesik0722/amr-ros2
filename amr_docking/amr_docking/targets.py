"""Docking target geometry: planar poses (x, y, yaw) and marker-based target estimation."""

import math

import yaml


def load_targets(path):
    with open(path) as f:
        return yaml.safe_load(f)


def compose(a, b):
    """Return pose b (given in frame a) expressed in a's parent frame."""
    ax, ay, ayaw = a
    bx, by, byaw = b
    c, s = math.cos(ayaw), math.sin(ayaw)
    return (ax + c * bx - s * by, ay + s * bx + c * by, wrap(ayaw + byaw))


def wrap(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def target_from_markers(markers, offsets):
    """Estimate the target frame from detected markers.

    markers: {id: (x, y, nx, ny)} marker centre and face normal (pointing out of the
             face, towards the camera), projected onto the robot's ground plane.
    offsets: {id: y} marker y offset in the target frame.
    Returns (x, y, yaw) of the target frame, or None.

    With two markers the line between them gives the orientation (robust); with one,
    its face normal is used.
    """
    seen = [i for i in markers if i in offsets]
    if not seen:
        return None
    if len(seen) >= 2:
        a = max(seen, key=lambda i: offsets[i])
        b = min(seen, key=lambda i: offsets[i])
        dy = offsets[a] - offsets[b]
        if dy <= 0.0:
            return None
        ux, uy = markers[a][0] - markers[b][0], markers[a][1] - markers[b][1]
        norm = math.hypot(ux, uy)
        if norm < 1e-6:
            return None
        ux, uy = ux / norm, uy / norm            # target +y
        xx, xy = uy, -ux                          # target +x
        ox = 0.5 * (markers[a][0] - offsets[a] * ux + markers[b][0] - offsets[b] * ux)
        oy = 0.5 * (markers[a][1] - offsets[a] * uy + markers[b][1] - offsets[b] * uy)
    else:
        i = seen[0]
        x, y, nx, ny = markers[i]
        norm = math.hypot(nx, ny)
        if norm < 1e-6:
            return None
        xx, xy = nx / norm, ny / norm
        ux, uy = -xy, xx
        ox, oy = x - offsets[i] * ux, y - offsets[i] * uy
    return (ox, oy, math.atan2(xy, xx))
