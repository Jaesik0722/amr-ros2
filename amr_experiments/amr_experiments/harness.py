"""Harness for repeated trials in the Isaac Sim world.

The reset is the only simulation-specific part: /sim/set_state teleports the robot and
the cart so that every trial starts from an exact, repeatable pose. Everything a trial
then measures comes from the same interfaces the real robot has (/odom, /joint_states,
/cart/latched, /battery_state) and the trial drives the same nodes (amr_docking
commands, Nav2), so a trial script also runs on hardware once the reset is replaced by
placing the robot by hand.

Ground truth: in simulation /odom is the planar ground-truth pose, so a docking error
is measured against the pose the target geometry implies, not against the robot's own
estimate.
"""

import math
import subprocess
import time

from amr_docking.targets import wrap
from geometry_msgs.msg import Pose, PoseArray, PoseStamped, PoseWithCovarianceStamped
from nav2_msgs.action import NavigateToPose
from nav_msgs.msg import Odometry
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import BatteryState, JointState
from std_msgs.msg import Bool
from std_srvs.srv import Empty

BEST_EFFORT = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)


def yaw_of(q):
    return 2.0 * math.atan2(q.z, q.w)


def pose_msg(x, y, yaw):
    p = Pose()
    p.position.x, p.position.y = float(x), float(y)
    p.orientation.z, p.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)
    return p


def pose_error(actual, expected):
    """(longitudinal, lateral, yaw) error of actual w.r.t. expected, in expected's frame."""
    dx, dy = actual[0] - expected[0], actual[1] - expected[1]
    c, s = math.cos(expected[2]), math.sin(expected[2])
    return (c * dx + s * dy, -s * dx + c * dy, wrap(actual[2] - expected[2]))


class Harness(Node):
    """One node per experiment: resets the world, reads state, runs the robot's commands."""

    def __init__(self, name='amr_experiment'):
        super().__init__(name, parameter_overrides=[Parameter('use_sim_time', value=True)])
        self.odom = None
        self.joints = {}
        self.latched = False
        self.battery = None
        self.cart = None
        self.samples = None

        self.pub_state = self.create_publisher(PoseArray, '/sim/set_state', 1)
        self.pub_initial = self.create_publisher(PoseWithCovarianceStamped, '/initialpose', 1)
        self.create_subscription(Odometry, '/odom', self._on_odom, 10)
        self.create_subscription(JointState, '/joint_states', self._on_joints, 10)
        self.create_subscription(Bool, '/cart/latched', self._on_latched, 10)
        self.create_subscription(BatteryState, '/battery_state', self._on_battery, 10)
        self.create_subscription(PoseStamped, '/indoor_gps/cart', self._on_cart, 10)
        self.nav = ActionClient(self, NavigateToPose, 'navigate_to_pose')

    # ---- state ---------------------------------------------------------------
    def _on_odom(self, msg):
        p = msg.pose.pose
        self.odom = (p.position.x, p.position.y, yaw_of(p.orientation))
        if self.samples is not None:
            cart = self.cart or (float('nan'),) * 3
            self.samples.append((self.time(), *self.odom, self.beta(), self.pitch_effort(),
                                 *cart))

    def _on_joints(self, msg):
        for i, n in enumerate(msg.name):
            self.joints[n] = (msg.position[i] if i < len(msg.position) else 0.0,
                              msg.effort[i] if i < len(msg.effort) else 0.0)

    def _on_latched(self, msg):
        self.latched = msg.data

    def _on_battery(self, msg):
        self.battery = (msg.percentage, msg.current, msg.power_supply_status)

    def _on_cart(self, msg):
        """Store the measured cart pose.

        Noisy like the real indoor GPS, but independent of the hitch geometry, so it is
        what a clearance computed from the hitch angle is checked against.
        """
        p = msg.pose
        self.cart = (p.position.x, p.position.y, yaw_of(p.orientation))

    def beta(self):
        """Hitch yaw angle [rad] (the robot-cart angle while towing)."""
        return self.joints.get('hitch_yaw_joint', (0.0, 0.0))[0]

    def pitch_effort(self):
        return abs(self.joints.get('hitch_pitch_joint', (0.0, 0.0))[1])

    def time(self):
        return self.get_clock().now().nanoseconds * 1e-9

    # ---- control -------------------------------------------------------------
    def spin(self, seconds):
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            rclpy.spin_once(self, timeout_sec=0.02)

    def wait_clock(self, timeout=150.0):
        """/clock first: a use_sim_time node reads time 0 until it arrives."""
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            rclpy.spin_once(self, timeout_sec=0.1)
            if self.time() > 0.0 and self.odom is not None:
                return True
        return False

    def reset(self, robot, cart=None, settle=2.5):
        """Teleport, relocalise and let the physics settle. Returns the ground-truth pose."""
        msg = PoseArray()
        msg.header.frame_id = 'map'
        msg.poses = [pose_msg(*robot)] + ([pose_msg(*cart)] if cart else [])
        for _ in range(3):              # the simulator keeps only the last request
            self.pub_state.publish(msg)
            self.spin(0.1)
        end = time.monotonic() + 10.0
        while time.monotonic() < end:
            self.spin(0.1)
            if self.odom and math.hypot(self.odom[0] - robot[0], self.odom[1] - robot[1]) < 0.05:
                break
        self.set_initial_pose(robot)
        self.clear_costmaps()
        self.spin(settle)
        return self.odom

    def set_initial_pose(self, pose):
        msg = PoseWithCovarianceStamped()
        msg.header.frame_id = 'map'
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose.pose = pose_msg(*pose)
        msg.pose.covariance[0] = msg.pose.covariance[7] = 0.01
        msg.pose.covariance[35] = 0.01
        for _ in range(3):
            self.pub_initial.publish(msg)
            self.spin(0.2)

    def clear_costmaps(self):
        for name in ('/global_costmap/clear_entirely_global_costmap',
                     '/local_costmap/clear_entirely_local_costmap'):
            client = self.create_client(Empty, name)
            if client.wait_for_service(timeout_sec=2.0):
                future = client.call_async(Empty.Request())
                rclpy.spin_until_future_complete(self, future, timeout_sec=3.0)
            self.destroy_client(client)

    def navigate(self, x, y, yaw=0.0, timeout=180.0):
        """Send a Nav2 goal. Returns (success, seconds)."""
        if not self.nav.wait_for_server(timeout_sec=10.0):
            return False, 0.0
        goal = NavigateToPose.Goal()
        goal.pose.header.frame_id = 'map'
        goal.pose.header.stamp = self.get_clock().now().to_msg()
        goal.pose.pose = pose_msg(x, y, yaw)
        t0 = self.time()
        handle_future = self.nav.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, handle_future, timeout_sec=10.0)
        handle = handle_future.result()
        if handle is None or not handle.accepted:
            return False, 0.0
        result_future = handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future, timeout_sec=timeout)
        result = result_future.result()
        if result is None:
            handle.cancel_goal_async()
            return False, self.time() - t0
        return result.status == 4, self.time() - t0   # 4 = SUCCEEDED

    def run_command(self, executable, timeout=300.0, package='amr_docking', settle=1.5):
        """Run one of the robot's command-line nodes. Returns (ok, seconds, output).

        The settle is not cosmetic. These commands start with a NavigateToPose of their
        own, and a trial that fires one the instant the previous goal reported success
        catches the planner still finishing that path: the behaviour tree then gives up
        waiting for compute_path_to_pose to acknowledge the request and the dock aborts
        with FAILED_TO_STAGE. A real mission never chains two goals with no gap either.
        """
        self.spin(settle)
        t0 = self.time()
        cmd = ['ros2', 'run', package, executable, '--ros-args', '-p', 'use_sim_time:=true']
        try:
            done = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
            out = done.stdout + done.stderr
            ok = done.returncode == 0      # the commands exit non-zero on failure
        except subprocess.TimeoutExpired as e:
            out = (e.stdout or b'').decode() if isinstance(e.stdout, bytes) else (e.stdout or '')
            ok = False
        self.spin(0.5)
        return ok, self.time() - t0, out

    # ---- trajectory recording -------------------------------------------------
    def start_recording(self):
        self.samples = []

    def stop_recording(self):
        samples, self.samples = self.samples or [], None
        return samples
