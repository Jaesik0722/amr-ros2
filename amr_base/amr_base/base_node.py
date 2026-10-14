"""ROS 2 driver for the MD Robot dual-channel motor controller."""

import math
import time

from amr_base import md_protocol as md
from amr_base.diff_drive import DiffDrive, DiffDriveConfig
from geometry_msgs.msg import TransformStamped, Twist, TwistStamped
from nav_msgs.msg import Odometry
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import JointState
import serial
from tf2_ros import TransformBroadcaster

SETUP_WAIT_S = 1.0


class BaseNode(Node):

    def __init__(self):
        super().__init__('amr_base')

        self.declare_parameter('port', '/dev/ttyWheel')
        self.declare_parameter('baudrate', 19200)
        self.declare_parameter('configure_driver_on_start', True)
        self.declare_parameter('update_rate', 20.0)
        self.declare_parameter('reply_timeout', 0.04)
        self.declare_parameter('cmd_vel_timeout', 0.5)
        self.declare_parameter('use_stamped_cmd_vel', False)
        self.declare_parameter('max_linear_velocity', 0.5)
        self.declare_parameter('max_angular_velocity', 1.5)
        self.declare_parameter('wheel_radius', 0.1)
        self.declare_parameter('wheel_separation', 0.3742)
        self.declare_parameter('gear_ratio', 4.33)
        self.declare_parameter('number_of_poles', 20)
        self.declare_parameter('encoder_ppr', 1280)
        self.declare_parameter('left_sign', -1)
        self.declare_parameter('right_sign', 1)
        self.declare_parameter('odom_frame', 'odom')
        self.declare_parameter('base_frame', 'base_footprint')
        self.declare_parameter('publish_tf', True)
        self.declare_parameter('left_wheel_joint', 'left_wheel_joint')
        self.declare_parameter('right_wheel_joint', 'right_wheel_joint')
        self.declare_parameter('pose_covariance_diagonal', [0.001, 0.001, 1e6, 1e6, 1e6, 0.01])
        self.declare_parameter('twist_covariance_diagonal', [0.002, 0.001, 1e6, 1e6, 1e6, 0.02])

        p = self.get_parameter
        self.poles = p('number_of_poles').value
        self.drive = DiffDrive(DiffDriveConfig(
            wheel_radius=p('wheel_radius').value,
            wheel_separation=p('wheel_separation').value,
            gear_ratio=p('gear_ratio').value,
            counts_per_motor_rev=self.poles * 3,
            left_sign=p('left_sign').value,
            right_sign=p('right_sign').value,
        ))
        self.odom_frame = p('odom_frame').value
        self.base_frame = p('base_frame').value
        self.wheel_joints = [p('left_wheel_joint').value, p('right_wheel_joint').value]
        self.max_lin = p('max_linear_velocity').value
        self.max_ang = p('max_angular_velocity').value
        self.cmd_timeout = p('cmd_vel_timeout').value
        self.reply_timeout = p('reply_timeout').value
        self.pose_cov = p('pose_covariance_diagonal').value
        self.twist_cov = p('twist_covariance_diagonal').value

        self.port = serial.Serial(p('port').value, p('baudrate').value, timeout=0)
        self.parser = md.FrameParser()
        if p('configure_driver_on_start').value:
            self._configure_driver(p('encoder_ppr').value)
        self._write(md.torque_off())

        self.cmd = (0.0, 0.0)
        self.last_cmd_time = None
        self.missed_replies = 0

        if p('use_stamped_cmd_vel').value:
            self.create_subscription(
                TwistStamped, 'cmd_vel', lambda m: self._on_cmd(m.twist), 10)
        else:
            self.create_subscription(Twist, 'cmd_vel', self._on_cmd, 10)
        self.pub_odom = self.create_publisher(Odometry, 'odom', 10)
        self.pub_joints = self.create_publisher(JointState, 'joint_states', 10)
        self.pub_motor = self.create_publisher(JointState, '~/motor_state', 10)
        self.tf_broadcaster = TransformBroadcaster(self) if p('publish_tf').value else None

        self.create_timer(1.0 / p('update_rate').value, self._update)
        self.get_logger().info(f'amr_base is up on {self.port.port}.')

    def _write(self, frame: bytes):
        self.port.write(frame)
        self.port.flush()

    def _configure_driver(self, ppr: int):
        # Only strictly needed once per driver, kept for parity with the C++ node.
        self.get_logger().info('Configuring hall sensor and encoder resolution.')
        self._write(md.hall_type(self.poles // 2))
        time.sleep(SETUP_WAIT_S)
        self._write(md.encoder_ppr(ppr))
        time.sleep(SETUP_WAIT_S)
        # Drop the acknowledgements so they do not pollute later reads.
        self.port.reset_input_buffer()

    def _on_cmd(self, msg: Twist):
        self.cmd = (max(-self.max_lin, min(self.max_lin, msg.linear.x)),
                    max(-self.max_ang, min(self.max_ang, msg.angular.z)))
        self.last_cmd_time = self.get_clock().now()

    def _read_reply(self):
        deadline = time.monotonic() + self.reply_timeout
        while time.monotonic() < deadline:
            data = self.port.read(self.port.in_waiting or 1)
            if data:
                frame = self.parser.feed(data)
                if frame is not None:
                    return frame
            else:
                time.sleep(0.001)
        return None

    def _update(self):
        now = self.get_clock().now()
        linear, angular = self.cmd
        if (self.last_cmd_time is None
                or (now - self.last_cmd_time).nanoseconds * 1e-9 > self.cmd_timeout):
            linear, angular = 0.0, 0.0

        rpm1, rpm2 = self.drive.twist_to_motor_rpm(linear, angular)
        self._write(md.velocity_command(round(rpm1), round(rpm2)))

        data = self._read_reply()
        if data is None:
            self.missed_replies += 1
            self.get_logger().warn(
                f'No reply from MD driver ({self.missed_replies} consecutive).',
                throttle_duration_sec=2.0)
            return
        self.missed_replies = 0
        self.drive.update(data.position_ch1, data.position_ch2, data.rpm_ch1, data.rpm_ch2)
        self._publish(now, data)

    def _publish(self, now, data: md.MainData):
        d = self.drive
        stamp = now.to_msg()
        qz, qw = math.sin(d.theta / 2.0), math.cos(d.theta / 2.0)

        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = self.odom_frame
        odom.child_frame_id = self.base_frame
        odom.pose.pose.position.x = d.x
        odom.pose.pose.position.y = d.y
        odom.pose.pose.orientation.z = qz
        odom.pose.pose.orientation.w = qw
        odom.twist.twist.linear.x = d.linear
        odom.twist.twist.angular.z = d.angular
        for i in range(6):
            odom.pose.covariance[i * 7] = self.pose_cov[i]
            odom.twist.covariance[i * 7] = self.twist_cov[i]
        self.pub_odom.publish(odom)

        if self.tf_broadcaster is not None:
            tf = TransformStamped()
            tf.header.stamp = stamp
            tf.header.frame_id = self.odom_frame
            tf.child_frame_id = self.base_frame
            tf.transform.translation.x = d.x
            tf.transform.translation.y = d.y
            tf.transform.rotation.z = qz
            tf.transform.rotation.w = qw
            self.tf_broadcaster.sendTransform(tf)

        c = d.cfg
        to_rad_s = 2.0 * math.pi / 60.0 / c.gear_ratio
        wheel_vel = [c.left_sign * data.rpm_ch1 * to_rad_s, c.right_sign * data.rpm_ch2 * to_rad_s]

        joints = JointState()
        joints.header.stamp = stamp
        joints.name = self.wheel_joints
        joints.position = [d.left_angle, d.right_angle]
        joints.velocity = wheel_vel
        self.pub_joints.publish(joints)

        # Motor currents in Amperes in the effort field (not torque); useful
        # for estimating towing load.
        motor = JointState()
        motor.header.stamp = stamp
        motor.name = self.wheel_joints
        motor.velocity = wheel_vel
        motor.effort = [data.current_ch1, data.current_ch2]
        self.pub_motor.publish(motor)

    def shutdown(self):
        try:
            self._write(md.velocity_command(0, 0, md.RETURN_NONE))
            time.sleep(0.1)
            self._write(md.torque_off())
            time.sleep(0.1)
        finally:
            self.port.close()


def main(args=None):
    rclpy.init(args=args)
    node = BaseNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.shutdown()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
