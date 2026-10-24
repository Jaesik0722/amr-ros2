"""E2: repeatability of reverse docking to a cart and of the coupling that follows.

Jazzy's opennav_docking cannot drive a backward dock (see CLAUDE.md), so the robot uses
amr_docking's own reverse controller. A trial moves the cart to a known pose, drives the
robot to a start pose, runs `dock_cart`, and measures the final pose against the pose the
cart geometry implies, plus the hitch effort the command read before and after lifting
the hook.

Only the cart is teleported between trials; the robot navigates, so SLAM stays
consistent. On hardware the cart is pushed to a marked pose instead.

  ros2 run amr_experiments e2_cart_docking --out results/e2 --repeats 6
"""

import argparse
import math
import os
import random
import re
import sys

from ament_index_python.packages import get_package_share_directory
from amr_docking.targets import compose, load_targets
import rclpy
from rclpy.utilities import remove_ros_args

from .harness import Harness, pose_error
from .trial_log import format_summary, summarize, TrialLog

EFFORT_RE = re.compile(r'effort: ([0-9.]+) before, ([0-9.]+) after')
RELEASE_SETTLE = 8.0     # how long the latch may take to open [s]
FIELDS = ['cart_x', 'cart_y', 'cart_yaw', 'start_x', 'start_y', 'reached_start',
          'dock_time', 'err_long', 'err_lat', 'err_yaw', 'err_xy',
          'effort_free', 'effort_loaded', 'latched', 'release_ok', 'latched_after',
          'release_note', 'success', 'note']

# Cart poses: a spread of positions and headings in the free part of the warehouse.
CART_POSES = [(3.0, -7.5, math.pi / 2), (4.5, -6.0, math.pi / 3), (2.0, -5.0, math.pi),
              (5.5, -7.0, -math.pi / 2), (3.5, -3.0, 0.0)]


def parse_args(argv):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', default='results/e2')
    ap.add_argument('--repeats', type=int, default=6, help='trials per cart pose')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--start-radius', type=float, nargs=2, default=[1.5, 3.0],
                    help='the robot starts this far from the cart')
    return ap.parse_args(remove_ros_args(args=argv if argv is not None else sys.argv)[1:])


def start_pose(rng, cart):
    """Pick a start pose on an arc in front of the cart, far enough to need Nav2."""
    bearing = rng.uniform(-math.pi / 2, math.pi / 2)
    radius = rng.uniform(1.8, 3.0)
    x = cart[0] + radius * math.cos(cart[2] + bearing)
    y = cart[1] + radius * math.sin(cart[2] + bearing)
    return (x, y, rng.uniform(-math.pi, math.pi))


def main(argv=None):
    args = parse_args(argv)
    targets = load_targets(os.path.join(get_package_share_directory('amr_docking'),
                                        'config', 'docking_targets.yaml'))
    cfg = targets['cart']
    rclpy.init()
    node = Harness('e2_cart_docking')
    if not node.wait_clock():
        node.get_logger().error('No /clock: start the simulator first.')
        return 1
    log = TrialLog(args.out, 'e2_cart_docking', FIELDS)
    rng = random.Random(args.seed)
    try:
        for cart in CART_POSES:
            expected = compose(compose(cart, tuple(cfg['target_in_cart'])),
                               tuple(cfg['robot_pose']))
            for _ in range(args.repeats):
                run_trial(node, log, cart, expected, start_pose(rng, cart))
    except KeyboardInterrupt:
        pass
    finally:
        summary = summarize(log.rows, ['dock_time', 'err_long', 'err_lat', 'err_yaw', 'err_xy',
                                       'effort_free', 'effort_loaded'])
        log.close(summary)
        print(format_summary('E2 cart reverse docking', summary))
        print(f'  log: {log.path}')
        node.destroy_node()
        rclpy.try_shutdown()
    return 0


def run_trial(node, log, cart, expected, start):
    """Place the cart, drive to the start pose, dock, measure, release."""
    if node.latched:
        node.run_command('release_cart', timeout=180.0)
    node.reset(node.odom or start, cart, settle=2.0)      # cart only: the robot stays put
    reached, _ = node.navigate(*start, timeout=180.0)
    ok, seconds, out = node.run_command('dock_cart', timeout=480.0)
    final = node.odom or (float('nan'),) * 3
    err = pose_error(final, expected)
    match = EFFORT_RE.search(out)
    free, loaded = (float(match.group(1)), float(match.group(2))) if match else ('', '')
    latched = node.latched
    release_ok, release_note = False, ''
    if latched:
        release_ok, _, release_out = node.run_command('release_cart', timeout=240.0)
        # Kept for every trial, not only the failures: a release that reports success
        # and leaves the cart on the hook is the interesting one, and without the
        # command's own last word there is nothing to tell the two apart afterwards.
        release_note = (release_out.strip().splitlines() or ['no output'])[-1][:90]
        # The command returns once the hook has been told to lower; the latch only opens
        # when it has physically lowered, and /cart/latched is published at a few Hz.
        # Reading the flag the instant the command returns measures the publish rate
        # rather than the release, so wait for it, bounded.
        deadline = node.time() + RELEASE_SETTLE
        while node.latched and node.time() < deadline:
            node.spin(0.25)
    note = '' if ok else out.strip().splitlines()[-1][:90] if out.strip() else 'no output'
    row = log.add(cart_x=f'{cart[0]:.3f}', cart_y=f'{cart[1]:.3f}', cart_yaw=f'{cart[2]:.3f}',
                  start_x=f'{start[0]:.3f}', start_y=f'{start[1]:.3f}',
                  reached_start=int(reached), dock_time=f'{seconds:.2f}',
                  err_long=f'{err[0]:.4f}', err_lat=f'{err[1]:.4f}', err_yaw=f'{err[2]:.4f}',
                  err_xy=f'{math.hypot(err[0], err[1]):.4f}',
                  effort_free=free, effort_loaded=loaded, latched=int(latched),
                  release_ok=int(release_ok), latched_after=int(node.latched),
                  release_note=release_note, success=ok, note=note)
    n = row['trial']
    node.get_logger().info(
        f'trial {n:3d}  cart ({cart[0]:.1f}, {cart[1]:.1f})  ok {int(ok)}  t {seconds:.1f} s'
        f'  err long {err[0]:+.3f} lat {err[1]:+.3f} yaw {math.degrees(err[2]):+.2f} deg'
        f'  effort {loaded}')
