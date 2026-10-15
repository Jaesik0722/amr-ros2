import math
import os

from amr_capture.crop_operator import inside_box
from amr_capture.export_dataset import (export, homogeneous, matrix_to_quat, quat_to_matrix,
                                        read_pcd, rpy_to_matrix, Trajectory)
import numpy as np
import pytest

CFG = {'image_topic': '/cam/image/compressed', 'camera_info_topic': '/cam/info',
       'pose_topic': '/Odometry',
       'T_rig_camera': [0.05, 0.0, 0.1, -math.pi / 2, 0.0, -math.pi / 2],
       'time_offset': 0.0, 'min_translation': 0.05, 'min_rotation_deg': 3.0,
       'blur_reject_ratio': 0.5, 'max_points': 1000}


def test_quaternion_round_trip():
    R = rpy_to_matrix(0.3, -0.7, 2.5)
    assert np.allclose(quat_to_matrix(matrix_to_quat(R)), R)


def test_default_camera_looks_forward():
    R = rpy_to_matrix(*CFG['T_rig_camera'][3:])
    assert np.allclose(R @ [0, 0, 1], [1, 0, 0])     # optical axis = rig +x
    assert np.allclose(R @ [0, 1, 0], [0, 0, -1])    # image down = rig -z


def test_trajectory_interpolation():
    q0 = [0, 0, 0, 1]
    q1 = [0, 0, math.sin(0.5), math.cos(0.5)]         # yaw 1 rad
    traj = Trajectory([(0.0, [0, 0, 0], q0), (1.0, [1, 2, 0], q1)])
    T = traj.at(0.5, max_gap=2.0)
    assert np.allclose(T[:3, 3], [0.5, 1.0, 0.0])
    assert math.isclose(math.atan2(T[1, 0], T[0, 0]), 0.5, abs_tol=1e-9)
    assert traj.at(-0.1) is None and traj.at(1.1) is None
    assert Trajectory([(0.0, [0, 0, 0], q0), (1.0, [0, 0, 0], q0)]).at(0.5, max_gap=0.5) is None


def test_inside_box():
    pts = [[-0.5, 0, 0], [0.5, 0, 0], [-0.5, 0.9, 0]]
    assert list(inside_box(pts, [-1.6, -0.6, -1.5], [-0.15, 0.6, 1.0])) == [True, False, False]


def test_read_pcd_binary(tmp_path):
    pts = np.array([[1, 2, 3, 7], [4, 5, 6, 8]], np.float32)
    path = tmp_path / 'm.pcd'
    path.write_bytes(b'# .PCD v0.7\nVERSION 0.7\nFIELDS x y z intensity\nSIZE 4 4 4 4\n'
                     b'TYPE F F F F\nCOUNT 1 1 1 1\nWIDTH 2\nHEIGHT 1\n'
                     b'VIEWPOINT 0 0 0 1 0 0 0\nPOINTS 2\nDATA binary\n' + pts.tobytes())
    assert np.allclose(read_pcd(str(path)), pts[:, :3])


def write_bag(path, n=40):
    """Rig drives 2 m along x with the camera looking forward; noise images (sharp)."""
    rosbag2_py = pytest.importorskip('rosbag2_py')
    import cv2
    from nav_msgs.msg import Odometry
    from rclpy.serialization import serialize_message
    from sensor_msgs.msg import CameraInfo, CompressedImage
    w = rosbag2_py.SequentialWriter()
    w.open(rosbag2_py.StorageOptions(uri=path, storage_id='sqlite3'),
           rosbag2_py.ConverterOptions('cdr', 'cdr'))
    for i, (name, typ) in enumerate([('/cam/info', 'sensor_msgs/msg/CameraInfo'),
                                     ('/cam/image/compressed', 'sensor_msgs/msg/CompressedImage'),
                                     ('/Odometry', 'nav_msgs/msg/Odometry')]):
        w.create_topic(rosbag2_py.TopicMetadata(id=i, name=name, type=typ,
                                                serialization_format='cdr'))
    rng = np.random.default_rng(1)
    for i in range(n * 4):                      # odometry at 4x the image rate
        t = i * 0.025
        odom = Odometry()
        odom.header.stamp.sec, odom.header.stamp.nanosec = int(t), int((t % 1) * 1e9)
        odom.pose.pose.position.x = t * 0.6
        odom.pose.pose.orientation.w = 1.0
        w.write('/Odometry', serialize_message(odom), int(t * 1e9))
    for i in range(1, n - 1):
        t = i * 0.1
        info = CameraInfo(width=64, height=48)
        info.k = [50.0, 0, 32.0, 0, 50.0, 24.0, 0, 0, 1]
        info.header.stamp.sec, info.header.stamp.nanosec = int(t), int((t % 1) * 1e9)
        w.write('/cam/info', serialize_message(info), int(t * 1e9))
        img = CompressedImage(format='jpeg')
        img.header = info.header
        img.data = cv2.imencode('.jpg', rng.integers(0, 255, (48, 64, 3), np.uint8))[1].tobytes()
        w.write('/cam/image/compressed', serialize_message(img), int(t * 1e9))
    del w


def test_export_end_to_end(tmp_path):
    bag = str(tmp_path / 'bag')
    write_bag(bag)
    out = str(tmp_path / 'ds')
    frames = export(bag, out, CFG, log=lambda *_: None)
    # 0.06 m per image (0.6 m/s, 10 Hz) > min_translation: every image with a pose is kept.
    assert len(frames) == 38
    assert os.path.isfile(os.path.join(out, frames[0][0]))
    # images.txt holds cam_T_world; invert and compare with rig pose x T_rig_camera.
    lines = [ln for ln in open(os.path.join(out, 'sparse/0/images.txt'))
             if ln.strip() and not ln.startswith('#')]
    vals = [float(v) for v in lines[3].split()[1:8]]
    qw, qx, qy, qz, tx, ty, tz = vals
    cam_T_world = homogeneous(quat_to_matrix(np.array([qx, qy, qz, qw])), [tx, ty, tz])
    world_T_cam = np.linalg.inv(cam_T_world)
    assert np.allclose(world_T_cam, frames[3][1], atol=1e-6)
    assert np.allclose(world_T_cam[:3, :3] @ [0, 0, 1], [1, 0, 0], atol=1e-6)
    assert math.isclose(world_T_cam[2, 3], 0.1, abs_tol=1e-6)
    assert 'PINHOLE 64 48 50.0 50.0 32.0 24.0' in open(
        os.path.join(out, 'sparse/0/cameras.txt')).read()


def test_floor_alignment_levels_a_tilted_room():
    from amr_capture.export_dataset import floor_alignment, rpy_to_matrix, transform_points
    rng = np.random.default_rng(0)
    floor = np.c_[rng.uniform(-5, 5, (4000, 2)), np.zeros(4000)]
    wall = np.c_[np.full(1500, 5.0), rng.uniform(-5, 5, 1500), rng.uniform(0, 3, 1500)]
    room = np.vstack([floor, wall])
    # As FAST-LIO would see it: lidar 0.35 m up, frame tilted by a few degrees.
    T = homogeneous(rpy_to_matrix(0.05, -0.08, 0.7), [0.3, -0.2, -0.35])
    seen = transform_points(T, room)
    A = floor_alignment(seen)
    levelled = transform_points(A, seen)
    assert np.abs(levelled[:4000, 2]).max() < 1e-6
    assert levelled[4000:, 2].min() > -1e-6 and levelled[4000:, 2].max() < 3.0 + 1e-6
