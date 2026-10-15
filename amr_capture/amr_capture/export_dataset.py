"""Turn a capture bag + FAST-LIO trajectory into a Gaussian Splatting dataset.

    ros2 run amr_capture export_dataset BAG OUT [--poses POSES] [--pcd MAP.pcd]
        [--config capture.yaml]

BAG    rosbag2 folder with the ZED images and camera_info (capture.launch.py output).
POSES  FAST-LIO trajectory: a rosbag2 folder with nav_msgs/Odometry on export.pose_topic
       (record /Odometry while replaying), or a TUM text file (t x y z qx qy qz qw).
       Default: look for the pose topic in BAG itself.
MAP    FAST-LIO map (pcd_save_en) for the initial points; optional. With a map, the
       output frame is levelled so the floor is z = 0 (OUT/alignment.txt holds the
       transform; pass it to pcd_to_mesh so the collision mesh matches).

Writes OUT/images/*.jpg, a COLMAP text model OUT/sparse/0/{cameras,images,points3D}.txt
(3DGRUT, gsplat, Inria 3DGS) and OUT/transforms.json (nerfstudio). Frames are kept every
min_translation / min_rotation_deg of motion, and blurred ones (variance of the Laplacian
well below the running median) are dropped. Camera pose = rig pose (interpolated at the
image time + time_offset) x T_rig_camera.
"""

import argparse
import bisect
import json
import math
import os
import sys

import cv2
import numpy as np
import yaml


# ---- rotations -----------------------------------------------------------------
def quat_to_matrix(q):
    """(x, y, z, w) -> 3x3."""
    x, y, z, w = q / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def matrix_to_quat(R):
    """3x3 -> (x, y, z, w), w >= 0."""
    t = np.trace(R)
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        q = [(R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s, s / 4]
    else:
        i = int(np.argmax(np.diag(R)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = math.sqrt(1.0 + R[i, i] - R[j, j] - R[k, k]) * 2
        q = [0.0, 0.0, 0.0, (R[k, j] - R[j, k]) / s]
        q[i] = s / 4
        q[j] = (R[j, i] + R[i, j]) / s
        q[k] = (R[k, i] + R[i, k]) / s
    q = np.array(q)
    return q if q[3] >= 0 else -q


def rpy_to_matrix(roll, pitch, yaw):
    cr, sr, cp, sp, cy, sy = (math.cos(roll), math.sin(roll), math.cos(pitch),
                              math.sin(pitch), math.cos(yaw), math.sin(yaw))
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def slerp(q0, q1, a):
    q0, q1 = q0 / np.linalg.norm(q0), q1 / np.linalg.norm(q1)
    d = float(np.dot(q0, q1))
    if d < 0:
        q1, d = -q1, -d
    if d > 0.9995:
        q = q0 + a * (q1 - q0)
        return q / np.linalg.norm(q)
    th = math.acos(d)
    return (math.sin((1 - a) * th) * q0 + math.sin(a * th) * q1) / math.sin(th)


def homogeneous(R, t):
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, t
    return T


class Trajectory:
    """Time-stamped rig poses with linear / slerp interpolation."""

    def __init__(self, samples):
        samples = sorted(samples, key=lambda s: s[0])
        self.t = [s[0] for s in samples]
        self.p = [np.asarray(s[1], float) for s in samples]
        self.q = [np.asarray(s[2], float) for s in samples]

    def __len__(self):
        return len(self.t)

    def at(self, t, max_gap=0.25):
        """4x4 world_T_rig at time t, or None outside the trajectory / across gaps."""
        i = bisect.bisect_right(self.t, t)
        if i == 0 or i == len(self.t):
            return None
        t0, t1 = self.t[i - 1], self.t[i]
        if t1 - t0 > max_gap:
            return None
        a = (t - t0) / (t1 - t0) if t1 > t0 else 0.0
        p = (1 - a) * self.p[i - 1] + a * self.p[i]
        return homogeneous(quat_to_matrix(slerp(self.q[i - 1], self.q[i], a)), p)


# ---- input ---------------------------------------------------------------------
def stamp_sec(header):
    return header.stamp.sec + header.stamp.nanosec * 1e-9


def open_bag(path):
    import rosbag2_py
    storage = 'mcap' if any(f.endswith('.mcap') for f in os.listdir(path)) else 'sqlite3'
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=path, storage_id=storage),
                rosbag2_py.ConverterOptions('cdr', 'cdr'))
    types = {t.name: t.type for t in reader.get_all_topics_and_types()}
    return reader, types


def read_messages(path, topics):
    """Yield (topic, msg) for the given topics, in bag order."""
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message
    reader, types = open_bag(path)
    wanted = [t for t in topics if t in types]
    if not wanted:
        return
    reader.set_filter(rosbag2_py.StorageFilter(topics=wanted))
    classes = {t: get_message(types[t]) for t in wanted}
    while reader.has_next():
        topic, data, _ = reader.read_next()
        yield topic, deserialize_message(data, classes[topic])


def load_trajectory(source, topic):
    if os.path.isfile(source):          # TUM: t x y z qx qy qz qw
        rows = [r.split() for r in open(source) if r.strip() and not r.startswith('#')]
        return Trajectory([(float(r[0]), [float(v) for v in r[1:4]],
                            [float(v) for v in r[4:8]]) for r in rows])
    samples = []
    for _, msg in read_messages(source, [topic]):
        p, q = msg.pose.pose.position, msg.pose.pose.orientation
        samples.append((stamp_sec(msg.header), [p.x, p.y, p.z], [q.x, q.y, q.z, q.w]))
    return Trajectory(samples)


def decode_image(msg):
    if hasattr(msg, 'format'):          # CompressedImage
        return cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)
    ch = {'bgr8': 3, 'rgb8': 3, 'bgra8': 4, 'rgba8': 4, 'mono8': 1}[msg.encoding]
    img = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.step)
    img = img[:, :msg.width * ch].reshape(msg.height, msg.width, ch)
    code = {'rgb8': cv2.COLOR_RGB2BGR, 'bgra8': cv2.COLOR_BGRA2BGR,
            'rgba8': cv2.COLOR_RGBA2BGR, 'mono8': cv2.COLOR_GRAY2BGR}.get(msg.encoding)
    return cv2.cvtColor(img, code) if code is not None else img.copy()


def sharpness(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (gray.shape[1] // 2, gray.shape[0] // 2))
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def read_pcd(path):
    """Minimal PCD reader (ascii / binary, uncompressed): returns (N, 3) xyz."""
    with open(path, 'rb') as f:
        header = {}
        while True:
            line = f.readline().decode('ascii', 'ignore').strip()
            if not line or line.startswith('#'):
                continue
            key, *vals = line.split()
            header[key.upper()] = vals
            if key.upper() == 'DATA':
                break
        body = f.read()
    names, sizes = header['FIELDS'], [int(s) for s in header['SIZE']]
    types, counts = header['TYPE'], [int(c) for c in header.get('COUNT', ['1'] * len(names))]
    n = int(header['POINTS'][0])
    kind = header['DATA'][0]
    if kind == 'ascii':
        rows = np.array([r.split() for r in body.decode().splitlines() if r.strip()], float)
        cols, c = {}, 0
        for name, cnt in zip(names, counts):
            cols[name] = c
            c += cnt
        return rows[:, [cols['x'], cols['y'], cols['z']]]
    if kind != 'binary':
        raise ValueError(f'unsupported PCD data type {kind}')
    dtype = np.dtype([(name, {'F': 'f', 'I': 'i', 'U': 'u'}[t] + str(s), (cnt,))
                      for name, s, t, cnt in zip(names, sizes, types, counts)])
    data = np.frombuffer(body[:n * dtype.itemsize], dtype=dtype)
    return np.stack([data[k][:, 0] for k in ('x', 'y', 'z')], axis=1).astype(float)


def floor_alignment(points, iterations=500, tolerance=0.03, max_tilt_deg=30.0, seed=0):
    """4x4 transform that puts the floor (dominant near-horizontal low plane) at z = 0.

    RANSAC over the lower half of the map; the result keeps the yaw of the input frame.
    """
    pts = np.asarray(points, float)
    rng = np.random.default_rng(seed)
    if len(pts) > 200000:                                  # FAST-LIO maps: millions of points
        pts = pts[rng.choice(len(pts), 200000, replace=False)]
    low = pts[pts[:, 2] <= np.median(pts[:, 2])]
    best, best_count = None, -1
    cos_tilt = math.cos(math.radians(max_tilt_deg))
    for _ in range(iterations):
        a, b, c = low[rng.choice(len(low), 3, replace=False)]
        n = np.cross(b - a, c - a)
        if np.linalg.norm(n) < 1e-9:
            continue
        n /= np.linalg.norm(n)
        if n[2] < 0:
            n = -n
        if n[2] < cos_tilt:
            continue
        count = int(np.sum(np.abs((low - a) @ n) < tolerance))
        if count > best_count:
            best, best_count = (n, a), count
    if best is None:
        raise RuntimeError('no floor plane found')
    n, a = best
    inliers = low[np.abs((low - a) @ n) < tolerance]      # refine with least squares
    centroid = inliers.mean(axis=0)
    n = np.linalg.svd(inliers - centroid, full_matrices=False)[2][2]
    n = n if n[2] > 0 else -n
    axis = np.cross(n, [0.0, 0.0, 1.0])                    # rotate n onto +z
    s, c = np.linalg.norm(axis), float(n[2])
    R = np.eye(3)
    if s > 1e-9:
        k = axis / s
        K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
        R = np.eye(3) + s * K + (1 - c) * K @ K
    T = homogeneous(R, [0.0, 0.0, 0.0])
    T[2, 3] = -(R @ centroid)[2]
    return T


def transform_points(T, points):
    return np.asarray(points) @ T[:3, :3].T + T[:3, 3]


# ---- output --------------------------------------------------------------------
def write_colmap(out, intrinsics, frames, points):
    w, h, fx, fy, cx, cy, dist = intrinsics
    d = os.path.join(out, 'sparse', '0')
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, 'cameras.txt'), 'w') as f:
        f.write('# CAMERA_ID MODEL WIDTH HEIGHT PARAMS[]\n')
        if any(abs(v) > 1e-9 for v in dist[:4]):
            k1, k2, p1, p2 = dist[:4]
            f.write(f'1 OPENCV {w} {h} {fx} {fy} {cx} {cy} {k1} {k2} {p1} {p2}\n')
        else:
            f.write(f'1 PINHOLE {w} {h} {fx} {fy} {cx} {cy}\n')
    with open(os.path.join(d, 'images.txt'), 'w') as f:
        f.write('# IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME\n# POINTS2D[]\n')
        for i, (name, world_T_cam) in enumerate(frames, 1):
            cam_T_world = np.linalg.inv(world_T_cam)
            x, y, z, qw = matrix_to_quat(cam_T_world[:3, :3])
            t = cam_T_world[:3, 3]
            f.write(f'{i} {qw} {x} {y} {z} {t[0]} {t[1]} {t[2]} 1 {name}\n\n')
    with open(os.path.join(d, 'points3D.txt'), 'w') as f:
        f.write('# POINT3D_ID X Y Z R G B ERROR TRACK[]\n')
        for i, (x, y, z) in enumerate(points, 1):
            f.write(f'{i} {x:.4f} {y:.4f} {z:.4f} 128 128 128 0\n')


def write_transforms(out, intrinsics, frames):
    """Write a nerfstudio transforms.json (camera-to-world, OpenGL axes)."""
    w, h, fx, fy, cx, cy, dist = intrinsics
    flip = np.diag([1.0, -1.0, -1.0, 1.0])
    data = {'camera_model': 'OPENCV', 'w': w, 'h': h, 'fl_x': fx, 'fl_y': fy, 'cx': cx,
            'cy': cy, 'k1': dist[0], 'k2': dist[1], 'p1': dist[2], 'p2': dist[3],
            'ply_file_path': 'sparse/0/points3D.ply',
            'frames': [{'file_path': name, 'transform_matrix': (T @ flip).tolist()}
                       for name, T in frames]}
    with open(os.path.join(out, 'transforms.json'), 'w') as f:
        json.dump(data, f, indent=1)


def write_ply(path, points):
    with open(path, 'w') as f:
        f.write('ply\nformat ascii 1.0\n'
                f'element vertex {len(points)}\nproperty float x\nproperty float y\n'
                'property float z\nproperty uchar red\nproperty uchar green\n'
                'property uchar blue\nend_header\n')
        for x, y, z in points:
            f.write(f'{x:.4f} {y:.4f} {z:.4f} 128 128 128\n')


# ---- main ----------------------------------------------------------------------
def export(bag, out, cfg, poses=None, pcd=None, align_floor=True, log=print):
    traj = load_trajectory(poses or bag, cfg['pose_topic'])
    if len(traj) < 2:
        raise RuntimeError(f'no trajectory on {cfg["pose_topic"]} in {poses or bag}')
    x, y, z, r, p, yw = cfg['T_rig_camera']
    rig_T_cam = homogeneous(rpy_to_matrix(r, p, yw), [x, y, z])
    os.makedirs(os.path.join(out, 'images'), exist_ok=True)
    intrinsics, frames, recent = None, [], []
    last = None
    stats = {'images': 0, 'no_pose': 0, 'still': 0, 'blurred': 0}
    min_rot = math.radians(cfg['min_rotation_deg'])
    for topic, msg in read_messages(bag, [cfg['camera_info_topic'], cfg['image_topic']]):
        if topic == cfg['camera_info_topic']:
            if intrinsics is None:
                k, dist = list(msg.k), list(msg.d) + [0.0] * 5
                intrinsics = (msg.width, msg.height, k[0], k[4], k[2], k[5], dist[:5])
            continue
        stats['images'] += 1
        rig = traj.at(stamp_sec(msg.header) + cfg['time_offset'])
        if rig is None:
            stats['no_pose'] += 1
            continue
        cam = rig @ rig_T_cam
        if last is not None:
            dt = np.linalg.norm(cam[:3, 3] - last[:3, 3])
            cos = (np.trace(last[:3, :3].T @ cam[:3, :3]) - 1) / 2
            if dt < cfg['min_translation'] and math.acos(max(-1.0, min(1.0, cos))) < min_rot:
                stats['still'] += 1
                continue
        img = decode_image(msg)
        s = sharpness(img)
        recent = (recent + [s])[-30:]
        if len(recent) >= 5 and s < cfg['blur_reject_ratio'] * float(np.median(recent)):
            stats['blurred'] += 1
            continue
        name = f'images/{len(frames):05d}.jpg'
        cv2.imwrite(os.path.join(out, name), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        frames.append((name, cam))
        last = cam
    if intrinsics is None:
        raise RuntimeError(f'no {cfg["camera_info_topic"]} in {bag}')
    points = np.zeros((0, 3))
    align = np.eye(4)
    if pcd:
        points = read_pcd(pcd)
        if align_floor:
            align = floor_alignment(points)
            points = transform_points(align, points)
            frames = [(name, align @ T) for name, T in frames]
        if len(points) > cfg['max_points']:
            idx = np.random.default_rng(0).choice(len(points), cfg['max_points'], replace=False)
            points = points[idx]
    write_colmap(out, intrinsics, frames, points)
    write_ply(os.path.join(out, 'sparse', '0', 'points3D.ply'), points)
    write_transforms(out, intrinsics, frames)
    np.savetxt(os.path.join(out, 'alignment.txt'), align,
               header='map (FAST-LIO camera_init) -> dataset frame, floor at z = 0')
    log(f'{len(frames)} frames from {stats["images"]} images (no pose {stats["no_pose"]}, '
        f'not moved {stats["still"]}, blurred {stats["blurred"]}); {len(points)} points '
        f'-> {out}')
    return frames


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('bag')
    ap.add_argument('out')
    ap.add_argument('--poses', help='FAST-LIO odometry bag or TUM file (default: BAG)')
    ap.add_argument('--pcd', help='FAST-LIO map for points3D')
    ap.add_argument('--config', help='capture.yaml (default: the installed one)')
    ap.add_argument('--no-align', action='store_true',
                    help='keep the FAST-LIO frame (default: floor fitted to z = 0 using --pcd)')
    args = ap.parse_args(argv)
    path = args.config
    if path is None:
        from ament_index_python.packages import get_package_share_directory
        path = os.path.join(get_package_share_directory('amr_capture'), 'config',
                            'capture.yaml')
    with open(path) as f:
        cfg = yaml.safe_load(f)['export']
    try:
        export(args.bag, args.out, cfg, args.poses, args.pcd, not args.no_align)
    except RuntimeError as exc:
        print(f'export_dataset: {exc}', file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
