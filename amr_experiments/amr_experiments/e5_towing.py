"""E5: how the towed cart follows the robot.

A trial couples the cart, drives a fixed route with corners, and records the robot pose,
the hitch angle and the hitch effort at the odometry rate. From that it computes

  off-tracking  - how far the cart centre cuts inside the path the robot drove, the
                  quantity that decides whether a towed robot fits through a corner;
  beta          - the robot-cart angle, whose range says how much the passive hitch
                  swings and how much an active hitch would have to correct;
  goal error    - where the robot stopped relative to the goal it was given.

The same route is driven with the cart and without it, so the cost of towing is measured
against the robot's own baseline rather than in absolute terms.

  ros2 run amr_experiments e5_towing --out results/e5 --repeats 3
"""

import argparse
import math
import os
import sys

from ament_index_python.packages import get_package_share_directory
from amr_docking.cart_towing import cart_pose_in_base
from amr_docking.targets import load_targets
import rclpy
from rclpy.utilities import remove_ros_args

from .harness import Harness, pose_error
from .trial_log import format_summary, summarize, TrialLog

FIELDS = ['towing', 'cart_mass', 'route_time', 'path_length', 'goal_err_xy', 'goal_err_yaw',
          'offtrack_max', 'offtrack_rms', 'beta_max', 'beta_rms', 'effort_mean',
          'aborted_goals', 'success']

# A route with a long straight, a right-angle corner and a sweeping turn. Only the
# positions are fixed: each goal heading is the bearing of the leg that reaches it.
#
# A towed goal heading that differs from the direction of travel asks the robot to turn
# on the spot, and with an off-axle hitch that drags the cart sideways until the passive
# yaw joint reaches its limit and the pair jackknifes. That is a property of the system,
# not of this route, so it is measured on its own (see the in-place rotation note in
# docs) instead of being hidden inside every towing trial.
ROUTE_POINTS = [(6.5, -6.0), (7.0, -2.5), (3.5, -2.0), (2.5, -5.5)]


def route_from(start):
    """Goals through ROUTE_POINTS, each facing the way the robot arrives."""
    goals, previous = [], (start[0], start[1])
    for point in ROUTE_POINTS:
        heading = math.atan2(point[1] - previous[1], point[0] - previous[0])
        goals.append((point[0], point[1], heading))
        previous = point
    return goals


def parse_args(argv):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', default='results/e5')
    ap.add_argument('--repeats', type=int, default=3)
    ap.add_argument('--cart-mass', type=float, default=22.0, help='mass set in the simulator')
    ap.add_argument('--cart-pose', type=float, nargs=3, default=[3.0, -7.5, math.pi / 2])
    ap.add_argument('--no-baseline', action='store_true',
                    help='skip the runs without the cart')
    return ap.parse_args(remove_ros_args(args=argv if argv is not None else sys.argv)[1:])


def transform(pose, local):
    c, s = math.cos(pose[2]), math.sin(pose[2])
    return (pose[0] + c * local[0] - s * local[1], pose[1] + s * local[0] + c * local[1])


def point_to_path(point, path):
    """Shortest distance from a point to a polyline."""
    best = float('inf')
    for (x0, y0), (x1, y1) in zip(path, path[1:]):
        dx, dy = x1 - x0, y1 - y0
        length = dx * dx + dy * dy
        t = 0.0 if length == 0 else max(0.0, min(1.0, ((point[0] - x0) * dx +
                                                       (point[1] - y0) * dy) / length))
        best = min(best, math.hypot(point[0] - (x0 + t * dx), point[1] - (y0 + t * dy)))
    return best


def analyse(samples, tow_bar_x, towing):
    """Off-tracking and hitch statistics from one recorded route."""
    if len(samples) < 10:
        return {}
    robot_path = [(s[1], s[2]) for s in samples]
    length = sum(math.dist(a, b) for a, b in zip(robot_path, robot_path[1:]))
    # The cart centre trails the robot by this much, so off-tracking only means
    # something once the robot has driven at least that far: before that the cart is
    # behind the start of the recorded path and the distance to it is not off-tracking.
    trailing = math.hypot(*cart_pose_in_base(0.0, tow_bar_x)[:2])
    travelled = 0.0
    offsets, betas = [], []
    for i, sample in enumerate(samples):
        x, y, yaw, beta = sample[1], sample[2], sample[3], sample[4]
        betas.append(beta)
        if i:
            travelled += math.dist(robot_path[i - 1], robot_path[i])
        if towing and travelled > trailing:
            centre = transform((x, y, yaw), cart_pose_in_base(beta, tow_bar_x)[:2])
            offsets.append(point_to_path(centre, robot_path))
    efforts = [s[5] for s in samples]
    out = {'path_length': length,
           'beta_max': max(abs(b) for b in betas),
           'beta_rms': math.sqrt(sum(b * b for b in betas) / len(betas)),
           'effort_mean': sum(efforts) / len(efforts)}
    if offsets:
        out['offtrack_max'] = max(offsets)
        out['offtrack_rms'] = math.sqrt(sum(o * o for o in offsets) / len(offsets))
    return out


def main(argv=None):
    args = parse_args(argv)
    targets = load_targets(os.path.join(get_package_share_directory('amr_docking'),
                                        'config', 'docking_targets.yaml'))
    tow_bar_x = float(targets['cart']['tow_bar_in_cart'][0])
    rclpy.init()
    node = Harness('e5_towing')
    if not node.wait_clock():
        node.get_logger().error('No /clock: start the simulator first.')
        return 1
    log = TrialLog(args.out, 'e5_towing', FIELDS)
    modes = [True] if args.no_baseline else [False, True]
    try:
        for towing in modes:
            for _ in range(args.repeats):
                run_trial(node, log, args, tow_bar_x, towing)
    except KeyboardInterrupt:
        pass
    finally:
        keys = ['route_time', 'path_length', 'goal_err_xy', 'offtrack_max', 'offtrack_rms',
                'beta_max', 'beta_rms', 'effort_mean']
        towed = [r for r in log.rows if int(r['towing']) == 1]
        solo = [r for r in log.rows if int(r['towing']) == 0]
        summary = {'towing': summarize(towed, keys), 'baseline': summarize(solo, keys)}
        log.close(summary)
        print(format_summary('E5 towing', summary['towing']))
        if solo:
            print(format_summary('E5 baseline (no cart)', summary['baseline']))
        print(f'  log: {log.path}')
        node.destroy_node()
        rclpy.try_shutdown()
    return 0


def run_trial(node, log, args, tow_bar_x, towing):
    """Couple (or not), drive the route while recording, release."""
    cart = tuple(args.cart_pose)
    if node.latched:
        node.run_command('release_cart', timeout=180.0)
    node.reset(node.odom or ROUTE_POINTS[-1] + (0.0,), cart, settle=2.0)
    coupled = False
    if towing:
        coupled, _, _ = node.run_command('dock_cart', timeout=480.0)
        if not coupled:
            log.add(towing=1, cart_mass=args.cart_mass, success=False)
            node.get_logger().warn('coupling failed; trial skipped')
            return
    route = route_from(node.odom or (0.0, 0.0, 0.0))
    node.start_recording()
    t0 = node.time()
    aborted = 0
    for goal in route:
        ok, _ = node.navigate(*goal, timeout=240.0)
        aborted += int(not ok)
    samples = node.stop_recording()
    seconds = node.time() - t0
    stats = analyse(samples, tow_bar_x, towing)
    err = pose_error(node.odom or (float('nan'),) * 3, route[-1])
    row = log.add(towing=int(towing), cart_mass=args.cart_mass,
                  route_time=f'{seconds:.2f}',
                  path_length=f"{stats.get('path_length', float('nan')):.3f}",
                  goal_err_xy=f'{math.hypot(err[0], err[1]):.4f}',
                  goal_err_yaw=f'{err[2]:.4f}',
                  offtrack_max=f"{stats.get('offtrack_max', float('nan')):.4f}",
                  offtrack_rms=f"{stats.get('offtrack_rms', float('nan')):.4f}",
                  beta_max=f"{stats.get('beta_max', float('nan')):.4f}",
                  beta_rms=f"{stats.get('beta_rms', float('nan')):.4f}",
                  effort_mean=f"{stats.get('effort_mean', float('nan')):.4f}",
                  aborted_goals=aborted, success=aborted == 0)
    log.save_trajectory(row['trial'], samples)
    if coupled:
        node.run_command('release_cart', timeout=240.0)
    n = row['trial']
    node.get_logger().info(
        f'trial {n:3d}  towing {int(towing)}  t {seconds:.1f} s  aborted {aborted}'
        f"  offtrack max {stats.get('offtrack_max', float('nan')):.3f} m"
        f"  beta max {math.degrees(stats.get('beta_max', 0.0)):.1f} deg")
