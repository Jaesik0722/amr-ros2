"""Towing mode: what changes while a cart hangs on the hitch.

Service /cart/set_coupled (SetBool), called by dock_cart / release_cart after the hitch
current confirmed the coupling. While coupled:
  - the Nav2 footprint (local and global costmap) becomes robot + cart, following the
    hitch yaw angle (convex hull, published at 5 Hz on <costmap>/footprint);
  - the cart is removed from the lidar cloud (livox/lidar -> livox/lidar_filtered, which
    the scan slice and costmaps use), so it is not seen as an obstacle;
  - the MPPI limits switch to towing values (slower, no reversing);
  - the hitch yaw becomes passive, so the cart trails like a trailer (optional).
The cloud filter always runs; it passes the cloud through while uncoupled.
"""

import math

from geometry_msgs.msg import Point32, Polygon
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.parameter_client import AsyncParameterClient
from rclpy.qos import DurabilityPolicy, qos_profile_sensor_data, QoSProfile
from rclpy.time import Time
from sensor_msgs.msg import JointState, PointCloud2
from std_msgs.msg import Bool
from std_srvs.srv import SetBool
from tf2_ros import Buffer, TransformException, TransformListener

ROBOT_FOOTPRINT = [[0.195, 0.07], [0.135, 0.14], [0.10, 0.215], [-0.10, 0.215],
                   [-0.135, 0.14], [-0.135, 0.04], [-0.28, 0.04], [-0.28, -0.04],
                   [-0.135, -0.04], [-0.135, -0.14], [-0.10, -0.215], [0.10, -0.215],
                   [0.135, -0.14], [0.195, -0.07]]
YAW_AXIS_X = -0.1768        # hitch yaw axis in base_footprint
TIP_FROM_YAW_X = -0.0918    # hook tip from the yaw axis (pitch axis + hook), level hook


def convex_hull(points):
    pts = sorted(set(map(tuple, points)))
    if len(pts) < 3:
        return pts

    def half(seq):
        out = []
        for p in seq:
            while len(out) >= 2 and ((out[-1][0] - out[-2][0]) * (p[1] - out[-2][1]) -
                                     (out[-1][1] - out[-2][1]) * (p[0] - out[-2][0])) <= 0:
                out.pop()
            out.append(p)
        return out
    lower, upper = half(pts), half(reversed(pts))
    return lower[:-1] + upper[:-1]


def cart_pose_in_base(hitch_yaw, tow_bar_x):
    """Cart frame (x towards the tow bar) in base_footprint, from the hitch yaw."""
    c, s = math.cos(hitch_yaw), math.sin(hitch_yaw)
    bar = (YAW_AXIS_X + c * TIP_FROM_YAW_X, s * TIP_FROM_YAW_X)
    return (bar[0] - c * tow_bar_x, bar[1] - s * tow_bar_x, hitch_yaw)


def cart_corners(pose, half_length, half_width, margin=0.0):
    x, y, yaw = pose
    c, s = math.cos(yaw), math.sin(yaw)
    hl, hw = half_length + margin, half_width + margin
    return [(x + c * px - s * py, y + s * px + c * py)
            for px, py in ((hl, hw), (-hl, hw), (-hl, -hw), (hl, -hw))]


class CartTowing(Node):

    def __init__(self):
        super().__init__('cart_towing')
        self.declare_parameter('tow_bar_x', 0.58)        # tow bar in the cart frame
        self.declare_parameter('cart_half_length', 0.42)
        self.declare_parameter('cart_half_width', 0.30)
        self.declare_parameter('filter_margin', 0.10)
        self.declare_parameter('base_frame', 'base_footprint')
        self.declare_parameter('passive_yaw', True)
        self.declare_parameter('controller', 'controller_server')
        self.declare_parameter('towing_limits', ['FollowPath.vx_max', '0.3',
                                                 'FollowPath.vx_min', '0.0',
                                                 'FollowPath.wz_max', '0.8'])
        g = self.get_parameter
        self.bar_x = g('tow_bar_x').value
        self.half = (g('cart_half_length').value, g('cart_half_width').value)
        self.margin = g('filter_margin').value
        self.base_frame = g('base_frame').value
        lim = list(g('towing_limits').value)
        self.towing_limits = {lim[i]: float(lim[i + 1]) for i in range(0, len(lim), 2)}
        self.normal_limits = {}
        self.coupled = False
        self.hitch_yaw = 0.0

        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.pub_coupled = self.create_publisher(Bool, '/cart/coupled', latched)
        self.pub_fp = [self.create_publisher(Polygon, f'/{c}/footprint', 10)
                       for c in ('local_costmap', 'global_costmap')]
        self.pub_cloud = self.create_publisher(PointCloud2, 'livox/lidar_filtered',
                                               qos_profile_sensor_data)
        self.create_subscription(PointCloud2, 'livox/lidar', self._on_cloud,
                                 qos_profile_sensor_data)
        self.create_subscription(JointState, 'joint_states', self._on_joints, 20)
        self.create_service(SetBool, '/cart/set_coupled', self._on_set_coupled)
        self.params = AsyncParameterClient(self, g('controller').value)
        self.yaw_client = self.create_client(SetBool, '/amr_hitch/set_yaw_passive')
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.create_timer(0.2, self._publish_footprint)
        self.pub_coupled.publish(Bool(data=False))

    # ---- state ----------------------------------------------------------------
    def _on_joints(self, msg):
        if 'hitch_yaw_joint' in msg.name:
            self.hitch_yaw = msg.position[msg.name.index('hitch_yaw_joint')]

    def _on_set_coupled(self, req, res):
        if req.data == self.coupled:
            res.success, res.message = True, 'unchanged'
            return res
        self.coupled = req.data
        self.pub_coupled.publish(Bool(data=self.coupled))
        self._set_limits(self.coupled)
        if self.get_parameter('passive_yaw').value and self.yaw_client.service_is_ready():
            self.yaw_client.call_async(SetBool.Request(data=self.coupled))
        self._publish_footprint(force=True)
        res.success = True
        res.message = 'towing mode' if self.coupled else 'normal mode'
        self.get_logger().info(res.message)
        return res

    def _set_limits(self, towing):
        if not self.params.services_are_ready():
            self.get_logger().warn('controller_server parameters unavailable; limits unchanged')
            return
        if towing and not self.normal_limits:
            future = self.params.get_parameters(list(self.towing_limits))
            future.add_done_callback(self._store_normal_limits)
            return
        values = self.towing_limits if towing else self.normal_limits
        self.params.set_parameters([Parameter(k, value=v) for k, v in values.items()])

    def _store_normal_limits(self, future):
        res = future.result()
        if res is None:
            return
        for name, value in zip(self.towing_limits, res.values):
            self.normal_limits[name] = value.double_value
        if self.coupled:
            self._set_limits(True)

    # ---- outputs --------------------------------------------------------------
    def _cart_pose(self):
        return cart_pose_in_base(self.hitch_yaw, self.bar_x)

    def _publish_footprint(self, force=False):
        if not self.coupled and not force:
            return
        pts = ROBOT_FOOTPRINT
        if self.coupled:
            pts = convex_hull(ROBOT_FOOTPRINT + cart_corners(self._cart_pose(), *self.half))
        poly = Polygon(points=[Point32(x=float(x), y=float(y)) for x, y in pts])
        for pub in self.pub_fp:
            pub.publish(poly)

    def _on_cloud(self, msg):
        if not self.coupled:
            self.pub_cloud.publish(msg)
            return
        try:
            tf = self.tf_buffer.lookup_transform(self.base_frame, msg.header.frame_id, Time())
        except TransformException as exc:
            self.get_logger().warn(f'TF: {exc}', throttle_duration_sec=5.0)
            return
        fields = {f.name: f.offset for f in msg.fields}
        raw = np.frombuffer(msg.data, np.uint8).reshape(-1, msg.point_step)
        xyz = np.stack([raw[:, fields[k]:fields[k] + 4].copy().view(np.float32)[:, 0]
                        for k in ('x', 'y', 'z')], axis=1)
        # The lidar is mounted level, so only the planar part of the transform matters.
        q = tf.transform.rotation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        c, s = math.cos(yaw), math.sin(yaw)
        bx = c * xyz[:, 0] - s * xyz[:, 1] + tf.transform.translation.x
        by = s * xyz[:, 0] + c * xyz[:, 1] + tf.transform.translation.y
        cx, cy, cyaw = self._cart_pose()
        cc, cs = math.cos(cyaw), math.sin(cyaw)
        lx = cc * (bx - cx) + cs * (by - cy)
        ly = -cs * (bx - cx) + cc * (by - cy)
        inside = ((np.abs(lx) <= self.half[0] + self.margin) &
                  (np.abs(ly) <= self.half[1] + self.margin))
        keep = raw[~inside]
        out = PointCloud2(header=msg.header, height=1, width=len(keep), fields=msg.fields,
                          is_bigendian=msg.is_bigendian, point_step=msg.point_step,
                          row_step=msg.point_step * len(keep), is_dense=msg.is_dense,
                          data=keep.tobytes())
        self.pub_cloud.publish(out)


def main():
    rclpy.init()
    node = CartTowing()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.try_shutdown()
