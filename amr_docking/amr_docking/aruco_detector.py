"""Detect a docking target's two ArUco markers and publish the robot's dock pose.

Subscribes a colour image + camera_info, estimates each marker's pose, projects the
markers onto the ground plane of base_footprint and publishes, on
`detected_dock_pose`, where base_footprint must be when docked (see
config/docking_targets.yaml). Feed it to opennav_docking with
use_external_detection_pose and zero external_detection_* offsets.
"""

import math
import os

from ament_index_python.packages import get_package_share_directory
import cv2
from geometry_msgs.msg import PoseStamped
import numpy as np
import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image
from tf2_ros import Buffer, TransformException, TransformListener

from .targets import compose, load_targets, target_from_markers


def quat_to_matrix(q):
    x, y, z, w = q.x, q.y, q.z, q.w
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def image_to_gray(msg):
    channels = {'rgb8': 3, 'bgr8': 3, 'rgba8': 4, 'bgra8': 4, 'mono8': 1}.get(msg.encoding)
    if channels is None:
        raise ValueError(f'unsupported encoding {msg.encoding}')
    buf = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.step)
    img = buf[:, :msg.width * channels].reshape(msg.height, msg.width, channels)
    if channels == 1:
        return img[:, :, 0]
    code = {'rgb8': cv2.COLOR_RGB2GRAY, 'bgr8': cv2.COLOR_BGR2GRAY,
            'rgba8': cv2.COLOR_RGBA2GRAY, 'bgra8': cv2.COLOR_BGRA2GRAY}[msg.encoding]
    return cv2.cvtColor(img, code)


class ArucoDockDetector(Node):

    def __init__(self):
        super().__init__('aruco_dock_detector')
        default_file = os.path.join(get_package_share_directory('amr_docking'), 'config',
                                    'docking_targets.yaml')
        self.declare_parameter('target', 'charger')
        self.declare_parameter('targets_file', default_file)
        self.declare_parameter('base_frame', 'base_footprint')
        self.declare_parameter('max_range', 3.0)
        cfg = load_targets(self.get_parameter('targets_file').value)
        name = self.get_parameter('target').value
        self.target = cfg[name]
        self.offsets = {int(k): float(v) for k, v in self.target['markers'].items()}
        self.robot_pose = tuple(self.target['robot_pose'])
        self.base_frame = self.get_parameter('base_frame').value
        self.max_range = self.get_parameter('max_range').value
        size = float(cfg['aruco']['marker_size'])
        h = size / 2.0
        self.object_points = np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]],
                                      np.float64)
        dictionary = cv2.aruco.getPredefinedDictionary(
            getattr(cv2.aruco, cfg['aruco']['dictionary']))
        if hasattr(cv2.aruco, 'ArucoDetector'):  # OpenCV >= 4.7
            detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())
            self.detect = detector.detectMarkers
        else:
            params = cv2.aruco.DetectorParameters_create()
            self.detect = lambda img: cv2.aruco.detectMarkers(img, dictionary, parameters=params)

        self.K = self.D = None
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.pub = self.create_publisher(PoseStamped, 'detected_dock_pose', 10)
        self.create_subscription(CameraInfo, 'camera_info', self._on_info, qos_profile_sensor_data)
        self.create_subscription(Image, 'image', self._on_image, qos_profile_sensor_data)
        self.get_logger().info(f'Looking for the {name} markers {sorted(self.offsets)}.')

    def _on_info(self, msg):
        self.K = np.array(msg.k, np.float64).reshape(3, 3)
        self.D = np.array(msg.d, np.float64) if len(msg.d) else np.zeros(5)

    def _on_image(self, msg):
        if self.K is None:
            return
        try:
            gray = image_to_gray(msg)
        except ValueError as exc:
            self.get_logger().error(str(exc), throttle_duration_sec=5.0)
            return
        corners, ids, _ = self.detect(gray)
        if ids is None:
            return
        try:
            tf = self.tf_buffer.lookup_transform(self.base_frame, msg.header.frame_id,
                                                 Time.from_msg(msg.header.stamp),
                                                 Duration(seconds=0.1))
        except TransformException as exc:
            self.get_logger().warn(f'TF: {exc}', throttle_duration_sec=5.0)
            return
        R = quat_to_matrix(tf.transform.rotation)
        t = np.array([tf.transform.translation.x, tf.transform.translation.y,
                      tf.transform.translation.z])
        markers = {}
        for c, i in zip(corners, ids.flatten()):
            if int(i) not in self.offsets:
                continue
            ok, rvec, tvec = cv2.solvePnP(self.object_points, c.reshape(4, 2).astype(np.float64),
                                          self.K, self.D, flags=cv2.SOLVEPNP_IPPE_SQUARE)
            if not ok or np.linalg.norm(tvec) > self.max_range:
                continue
            normal_cam = cv2.Rodrigues(rvec)[0][:, 2]   # out of the marker face
            p = R @ tvec.flatten() + t
            n = R @ normal_cam
            markers[int(i)] = (p[0], p[1], n[0], n[1])
        target = target_from_markers(markers, self.offsets)
        if target is None:
            return
        x, y, yaw = compose(target, self.robot_pose)
        out = PoseStamped()
        out.header.stamp = msg.header.stamp
        out.header.frame_id = self.base_frame
        out.pose.position.x, out.pose.position.y = x, y
        out.pose.orientation.z, out.pose.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)
        self.pub.publish(out)


def main():
    rclpy.init()
    node = ArucoDockDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.try_shutdown()
