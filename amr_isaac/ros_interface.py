"""ROS 2 side of the simulator, mirroring the real robot's topic interface.

Real robot (amr_base + amr_hitch)        Simulator (this node)
  sub  /cmd_vel                            same, same limits and timeout
  pub  /odom, /tf odom->base_footprint     same (ground truth, planar)
  pub  /joint_states (wheels, hitch)       same, hitch effort = joint torque [Nm]
  sub  /amr_hitch/command (JointState)     same
  srv  /amr_hitch/set_torque (SetBool)     same, torque off = passive hitch
  srv  /amr_hitch/set_yaw_passive          same, only the yaw joint free (towing)
  -                                        pub /clock (use_sim_time:=true)
  -                                        sub /sim/set_state (experiment reset)
"""

import math

from geometry_msgs.msg import PoseArray, TransformStamped, Twist
from nav_msgs.msg import Odometry
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import JointState
from std_srvs.srv import SetBool
from tf2_msgs.msg import TFMessage


def stamp(t):
    sec = int(t)
    return sec, int(round((t - sec) * 1e9))


class SimRosInterface:

    def __init__(self, cfg, hitch_joints):
        rclpy.init()
        self.node = rclpy.create_node('amr_isaac_sim')
        self.cfg = cfg
        self.hitch_joints = hitch_joints
        self.cmd = (0.0, 0.0)
        self.cmd_time = -1e9
        self.hitch_target = {name: 0.0 for name in hitch_joints}
        self.hitch_torque = True
        self.set_state = None       # pending teleport, applied by the main loop
        self.now = 0.0
        n = self.node
        n.create_subscription(Twist, 'cmd_vel', self._on_cmd, 10)
        n.create_subscription(JointState, 'amr_hitch/command', self._on_hitch, 10)
        n.create_subscription(PoseArray, 'sim/set_state', self._on_set_state, 1)
        n.create_service(SetBool, 'amr_hitch/set_torque', self._on_torque)
        n.create_service(SetBool, 'amr_hitch/set_yaw_passive', self._on_yaw_passive)
        self.yaw_passive = False
        self.pub_odom = n.create_publisher(Odometry, 'odom', 10)
        self.pub_tf = n.create_publisher(TFMessage, 'tf', 50)
        self.pub_joints = n.create_publisher(JointState, 'joint_states', 10)
        self.pub_clock = n.create_publisher(Clock, 'clock', 10)
        static_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE,
                                durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.pub_tf_static = n.create_publisher(TFMessage, 'tf_static', static_qos)
        # Spun from the sim loop: a spinning thread starves the sim of the GIL.
        self.executor = SingleThreadedExecutor()
        self.executor.add_node(n)

    # ---- inputs -------------------------------------------------------------
    def _on_set_state(self, msg):
        """Teleport for repeated experiments: poses[0] robot, poses[1] cart (optional).

        Not on the real robot. The main loop applies it between physics steps.
        """
        def planar(p):
            return (p.position.x, p.position.y,
                    2.0 * math.atan2(p.orientation.z, p.orientation.w))

        if msg.poses:
            self.set_state = (planar(msg.poses[0]),
                              planar(msg.poses[1]) if len(msg.poses) > 1 else None)

    def _on_cmd(self, msg):
        c = self.cfg
        v = max(-c['max_linear'], min(c['max_linear'], msg.linear.x))
        w = max(-c['max_angular'], min(c['max_angular'], msg.angular.z))
        self.cmd = (v, w)
        self.cmd_time = self.now

    def _on_hitch(self, msg):
        if not self.hitch_torque:
            self.node.get_logger().warn('Ignoring command, torque is off.',
                                        throttle_duration_sec=2.0)
            return
        if len(msg.position) != len(msg.name):
            self.node.get_logger().warn('Command needs one position per joint name.')
            return
        for name, pos in zip(msg.name, msg.position):
            if name in self.hitch_target:
                lo, hi = self.cfg['hitch_limits']
                self.hitch_target[name] = max(lo, min(hi, pos))

    def _on_torque(self, req, res):
        self.hitch_torque = req.data
        res.success = True
        res.message = 'torque ' + ('on' if req.data else 'off')
        return res

    def _on_yaw_passive(self, req, res):
        self.yaw_passive = req.data
        res.success = True
        res.message = 'yaw ' + ('passive' if req.data else 'holding')
        return res

    def wheel_targets(self):
        """Return wheel speeds [rad/s] for the latest cmd_vel, zero after the timeout."""
        v, w = self.cmd
        if self.now - self.cmd_time > self.cfg['cmd_vel_timeout']:
            v, w = 0.0, 0.0
        r, sep = self.cfg['wheel_radius'], self.cfg['wheel_separation']
        return (v - w * sep / 2.0) / r, (v + w * sep / 2.0) / r

    # ---- outputs ------------------------------------------------------------
    def publish_clock(self, t):
        msg = Clock()
        msg.clock.sec, msg.clock.nanosec = stamp(t)
        self.pub_clock.publish(msg)

    def publish_state(self, t, x, y, yaw, vx, wz, names, pos, vel, eff):
        sec, nsec = stamp(t)
        qz, qw = math.sin(yaw / 2.0), math.cos(yaw / 2.0)
        odom = Odometry()
        odom.header.stamp.sec, odom.header.stamp.nanosec = sec, nsec
        odom.header.frame_id = 'odom'
        odom.child_frame_id = 'base_footprint'
        odom.pose.pose.position.x, odom.pose.pose.position.y = x, y
        odom.pose.pose.orientation.z, odom.pose.pose.orientation.w = qz, qw
        odom.twist.twist.linear.x, odom.twist.twist.angular.z = vx, wz
        for i, var in ((0, 1e-3), (7, 1e-3), (14, 1e6), (21, 1e6), (28, 1e6), (35, 1e-3)):
            odom.pose.covariance[i] = var
            odom.twist.covariance[i] = var
        self.pub_odom.publish(odom)

        tf = TransformStamped()
        tf.header = odom.header
        tf.child_frame_id = 'base_footprint'
        tf.transform.translation.x, tf.transform.translation.y = x, y
        tf.transform.rotation.z, tf.transform.rotation.w = qz, qw
        self.pub_tf.publish(TFMessage(transforms=[tf]))

        js = JointState()
        js.header.stamp = odom.header.stamp
        js.name, js.position, js.velocity, js.effort = names, pos, vel, eff
        self.pub_joints.publish(js)

    def publish_static(self, transforms):
        """Publish static transforms, each (parent, child, xyz, quat_xyzw)."""
        msgs = []
        for parent, child, xyz, q in transforms:
            tf = TransformStamped()
            tf.header.frame_id, tf.child_frame_id = parent, child
            (tf.transform.translation.x, tf.transform.translation.y,
             tf.transform.translation.z) = xyz
            (tf.transform.rotation.x, tf.transform.rotation.y,
             tf.transform.rotation.z, tf.transform.rotation.w) = q
            msgs.append(tf)
        self.pub_tf_static.publish(TFMessage(transforms=msgs))

    def set_time(self, now):
        self.now = now

    def spin_once(self):
        """Handle at most one pending callback, without waiting."""
        self.executor.spin_once(timeout_sec=0.0)

    def shutdown(self):
        self.executor.shutdown()
        self.node.destroy_node()
        rclpy.try_shutdown()
