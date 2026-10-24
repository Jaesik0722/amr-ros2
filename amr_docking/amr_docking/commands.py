"""Command-line docking helpers.

Usage:
    ros2 run amr_docking dock_charger      # Nav2 docking_server, dock 'home_charger'
    ros2 run amr_docking undock_charger
    ros2 run amr_docking dock_cart         # cart pose from indoor GPS, back up, hook
    ros2 run amr_docking release_cart      # unhook, drive away from the cart

Cart docking: Nav2 brings the robot to a staging pose 0.8 m in front of the cart (from
the indoor GPS), then reverse_docking backs it onto the rear-camera ArUco estimate. The
hook is lowered on the way in and raised under the tow bar. The hitch pitch effort
(Dynamixel present current on the robot, joint torque in the simulator) tells whether
the bar is really in the hook: with a cart the hook carries the tongue load, without one
it moves freely.
"""

import math
import os
import sys
import time

from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import PoseStamped, Twist
from nav2_msgs.action import DockRobot, NavigateToPose, UndockRobot
import rclpy
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import JointState
from std_srvs.srv import SetBool
from tf2_ros import Buffer, TransformException, TransformListener

from .reverse_docking import reverse_command
from .targets import compose, load_targets

STAGING_DISTANCE = 0.8


def yaw_of(q):
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


def pose_msg(frame, pose):
    msg = PoseStamped()
    msg.header.frame_id = frame
    msg.pose.position.x, msg.pose.position.y = pose[0], pose[1]
    msg.pose.orientation.z, msg.pose.orientation.w = math.sin(pose[2] / 2), math.cos(pose[2] / 2)
    return msg


class Commander(Node):

    def __init__(self):
        super().__init__('amr_docking_cli')
        default_file = os.path.join(get_package_share_directory('amr_docking'), 'config',
                                    'docking_targets.yaml')
        self.declare_parameter('targets_file', default_file)
        self.declare_parameter('cart_pose_topic', '/indoor_gps/cart')
        self.declare_parameter('hook_down', -0.2)       # rad, hitch pitch while approaching
        self.declare_parameter('hook_up', 0.2)          # rad, hitch pitch that lifts the bar
        # |hitch pitch effort| above which the cart counts as hooked. Robot: present
        # current [A] (amr_hitch); simulator: torque [Nm]. Tune on the real hitch.
        self.declare_parameter('engage_effort', 0.5)
        self.declare_parameter('detection_topic', '/cart/detected_dock_pose')
        self.declare_parameter('odom_frame', 'odom')
        self.declare_parameter('base_frame', 'base_footprint')
        self.targets = load_targets(self.get_parameter('targets_file').value)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        # With use_sim_time the clock reads 0 until /clock arrives; wait for it.
        deadline = time.monotonic() + 10.0
        while self.get_clock().now().nanoseconds == 0 and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)

    def run_action(self, action_type, name, goal, timeout=180.0):
        client = ActionClient(self, action_type, name)
        if not client.wait_for_server(timeout_sec=10.0):
            self.get_logger().error(f'{name} is not available.')
            return None
        handle_future = client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, handle_future, timeout_sec=10.0)
        handle = handle_future.result()
        if handle is None or not handle.accepted:
            self.get_logger().error(f'{name}: goal rejected.')
            return None
        result_future = handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future, timeout_sec=timeout)
        if result_future.result() is None:
            self.get_logger().error(f'{name}: no result within {timeout:.0f} s.')
            return None
        return result_future.result().result

    def call_bool(self, name, value):
        client = self.create_client(SetBool, name)
        if not client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error(f'{name} is not available.')
            return False
        future = client.call_async(SetBool.Request(data=value))
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
        res = future.result()
        if res is None:
            return False
        self.get_logger().info(f'{name}: {res.message}')
        return res.success

    def wait(self, seconds):
        end = self.get_clock().now().nanoseconds + seconds * 1e9
        while self.get_clock().now().nanoseconds < end:
            rclpy.spin_once(self, timeout_sec=0.05)

    def set_hook(self, pitch, settle=1.5):
        if not hasattr(self, 'hitch_pub'):
            self.hitch_pub = self.create_publisher(JointState, '/amr_hitch/command', 10)
            self.wait(0.5)
        self.hitch_pub.publish(JointState(name=['hitch_yaw_joint', 'hitch_pitch_joint'],
                                          position=[0.0, pitch]))
        self.wait(settle)

    def hook_effort(self, seconds=0.5, attempts=3):
        """Mean |hitch pitch effort| over `seconds`, or None if nothing was measured.

        A reading that never arrived is not a reading of zero. This used to return 0.0
        when no joint state turned up in the window, which tells the release that the
        hook is free while it may still be carrying the whole tongue load: the robot
        then drives away from a cart that is still on the hitch and reports success.
        That is what left the latch closed in 2 of 30 E2 trials with the four-caster
        cart and once during a mission recording, both times under heavy simulator load
        where messages are late rather than absent.
        """
        for attempt in range(attempts):
            efforts = []

            def on_joints(msg):
                if 'hitch_pitch_joint' in msg.name and msg.effort:
                    efforts.append(msg.effort[msg.name.index('hitch_pitch_joint')])
            sub = self.create_subscription(JointState, '/joint_states', on_joints, 20)
            self.wait(seconds)
            self.destroy_subscription(sub)
            if efforts:
                return abs(sum(efforts) / len(efforts))
            self.get_logger().warn(f'No joint states while reading the hook effort '
                                   f'(attempt {attempt + 1}).')
        return None

    def planar_tf(self, target, source, stamp=None):
        tf = self.tf_buffer.lookup_transform(target, source, stamp or Time(),
                                             Duration(seconds=0.2))
        t = tf.transform
        return (t.translation.x, t.translation.y, yaw_of(t.rotation))

    def reverse_dock(self, timeout=60.0):
        """Back onto the rear-camera dock estimate (kept in odom when the markers drop out)."""
        odom, base = self.get_parameter('odom_frame').value, self.get_parameter('base_frame').value
        latest = {}

        def on_detection(msg):
            try:
                to_odom = self.planar_tf(odom, msg.header.frame_id,
                                         Time.from_msg(msg.header.stamp))
            except TransformException:
                return
            p = msg.pose
            latest['pose'] = compose(to_odom, (p.position.x, p.position.y, yaw_of(p.orientation)))
            latest['t'] = time.monotonic()
        sub = self.create_subscription(PoseStamped, self.get_parameter('detection_topic').value,
                                       on_detection, 10)
        start = time.monotonic()
        result = False
        try:
            while time.monotonic() - start < timeout:
                rclpy.spin_once(self, timeout_sec=0.05)
                if 'pose' not in latest:
                    if time.monotonic() - start > 5.0:
                        self.get_logger().error('The cart markers are not visible.')
                        return False
                    continue
                if time.monotonic() - latest['t'] > 5.0:
                    self.get_logger().error('Lost the cart markers.')
                    return False
                try:
                    bx, by, byaw = self.planar_tf(odom, base)
                except TransformException:
                    continue
                tx, ty, tyaw = latest['pose']
                c, s = math.cos(byaw), math.sin(byaw)
                target = (c * (tx - bx) + s * (ty - by), -s * (tx - bx) + c * (ty - by),
                          tyaw - byaw)
                v, w, remaining = reverse_command(target)
                if remaining <= 0.012:
                    result = True
                    break
                cmd = Twist()
                cmd.linear.x, cmd.angular.z = v, w
                self.cmd_pub.publish(cmd)
        finally:
            self.cmd_pub.publish(Twist())
            self.destroy_subscription(sub)
        return result

    def drive_straight(self, distance, speed=0.15):
        odom, base = self.get_parameter('odom_frame').value, self.get_parameter('base_frame').value
        start = None
        cmd = Twist()
        cmd.linear.x = math.copysign(speed, distance)
        deadline = time.monotonic() + abs(distance) / speed + 10.0
        while time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)
            try:
                x, y, _ = self.planar_tf(odom, base)
            except TransformException:
                continue
            start = start or (x, y)
            if math.hypot(x - start[0], y - start[1]) >= abs(distance):
                break
            self.cmd_pub.publish(cmd)
        self.cmd_pub.publish(Twist())

    def wait_cart_pose(self, timeout=5.0):
        msgs = []
        sub = self.create_subscription(PoseStamped, self.get_parameter('cart_pose_topic').value,
                                       msgs.append, 10)
        end = time.monotonic() + timeout
        while not msgs and time.monotonic() < end:
            rclpy.spin_once(self, timeout_sec=0.1)
        self.destroy_subscription(sub)
        return msgs[-1] if msgs else None


def _report(node, what, result):
    if result is None:
        return 1
    if result.success:
        node.get_logger().info(f'{what}: done.')
        return 0
    node.get_logger().error(f'{what}: failed, error {result.error_code} {result.error_msg}')
    return 1


def _main(fn):
    rclpy.init()
    node = Commander()
    try:
        code = fn(node)
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
    sys.exit(code)


def _dock_charger(node):
    goal = DockRobot.Goal(use_dock_id=True, dock_id='home_charger', navigate_to_staging_pose=True)
    return _report(node, 'dock charger', node.run_action(DockRobot, '/dock_robot', goal))


def _undock_charger(node):
    goal = UndockRobot.Goal(dock_type='charger_dock')
    return _report(node, 'undock charger', node.run_action(UndockRobot, '/undock_robot', goal))


def _dock_cart(node):
    cart = node.wait_cart_pose()
    if cart is None:
        node.get_logger().error('No cart pose from the indoor GPS.')
        return 1
    cfg = node.targets['cart']
    cart_pose = (cart.pose.position.x, cart.pose.position.y, yaw_of(cart.pose.orientation))
    dock = compose(compose(cart_pose, tuple(cfg['target_in_cart'])), tuple(cfg['robot_pose']))
    staging = compose(dock, (STAGING_DISTANCE, 0.0, 0.0))
    node.get_logger().info(f'Cart at ({cart_pose[0]:.2f}, {cart_pose[1]:.2f}); staging at '
                           f'({staging[0]:.2f}, {staging[1]:.2f}).')
    node.call_bool('/amr_hitch/set_yaw_passive', False)
    node.set_hook(0.0, settle=0.2)
    goal = NavigateToPose.Goal(pose=pose_msg(cart.header.frame_id, staging))
    for attempt in range(2):    # one retry: Nav2 sometimes aborts on a transient TF error
        result = node.run_action(NavigateToPose, '/navigate_to_pose', goal)
        if result is not None and result.error_code == 0:
            break
        node.get_logger().warn(f'Navigation to the staging pose failed (attempt {attempt + 1}).')
    if result is None or result.error_code != 0:
        node.get_logger().error('Could not reach the cart staging pose.')
        return 1
    node.set_hook(node.get_parameter('hook_down').value, settle=0.5)
    if not node.reverse_dock():
        node.drive_straight(STAGING_DISTANCE)
        return 1
    node.get_logger().info('At the tow bar.')
    free = node.hook_effort()
    node.set_hook(node.get_parameter('hook_up').value)
    loaded = node.hook_effort()
    threshold = node.get_parameter('engage_effort').value
    if loaded is None:
        node.get_logger().error('Could not read the hitch effort; treating the cart as '
                                'not hooked.')
        node.set_hook(node.get_parameter('hook_down').value)
        return 1
    node.get_logger().info(f'Hitch pitch effort: {free if free is None else f"{free:.2f}"} '
                           f'before, {loaded:.2f} after '
                           f'raising the hook (threshold {threshold:.2f}).')
    if loaded < threshold:
        node.get_logger().error('The tow bar is not in the hook; lowering it again.')
        node.set_hook(node.get_parameter('hook_down').value)
        return 1
    node.get_logger().info('Cart hooked.')
    return 0 if node.call_bool('/cart/set_coupled', True) else 1


def _release_cart(node):
    node.call_bool('/cart/set_coupled', False)
    threshold = node.get_parameter('engage_effort').value
    # The hook carries the tongue load until the tow bar has slid off it, so the servo
    # still reads as loaded for a moment after the hook starts going down. Heavily loaded
    # releases needed more than one settle: look again a few times before reporting that
    # the cart is still on the hitch, which is a fault an operator has to clear.
    for attempt in range(4):
        node.set_hook(node.get_parameter('hook_down').value)
        effort = node.hook_effort()
        if effort is not None and effort < threshold:
            break
        reading = 'nothing' if effort is None else f'{effort:.2f}'
        node.get_logger().info(f'The hook still reads {reading}; waiting for the load '
                               f'to come off (attempt {attempt + 1}).')
    else:
        # Unknown counts as loaded: driving away from a cart that is still on the hitch
        # is the failure this check exists to prevent.
        reading = 'nothing' if effort is None else f'{effort:.2f}'
        node.get_logger().error(f'The hook still reads {reading}; not driving away.')
        return 1
    node.drive_straight(STAGING_DISTANCE)
    node.set_hook(0.0, settle=0.5)
    node.get_logger().info('Left the cart.')
    return 0


def dock_charger():
    _main(_dock_charger)


def undock_charger():
    _main(_undock_charger)


def dock_cart():
    _main(_dock_cart)


def release_cart():
    _main(_release_cart)
