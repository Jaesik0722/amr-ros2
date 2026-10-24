"""E1: repeatability of forward docking to the wireless charger.

Each trial drives the robot to a random reachable start pose, runs the same
`dock_charger` command the robot runs, and measures where it ended up relative to the
pose the charger geometry implies. The robot is never teleported here: it navigates to
the start pose, so SLAM stays consistent and the trial is the whole pipeline
(Nav2 -> staging -> ArUco -> docking controller), not just the final approach.

Charging is reported as the simulator models it (receiver coil over the pad within a
fixed box), so the charging column says whether the final pose was inside that box - the
real alignment tolerance of the WiBotic pair has to be measured on hardware.

  ros2 run amr_experiments e1_charger_docking --out results/e1 --trials 30
"""

import argparse
import math
import os
import random
import sys

from ament_index_python.packages import get_package_share_directory
from amr_docking.targets import compose, load_targets
import rclpy
from rclpy.utilities import remove_ros_args

from .harness import Harness, pose_error
from .trial_log import format_summary, summarize, TrialLog

FIELDS = ['start_x', 'start_y', 'start_yaw', 'start_distance', 'reached_start',
          'dock_time', 'err_long', 'err_lat', 'err_yaw', 'err_xy',
          'charging', 'battery', 'undock_ok', 'success', 'note']


def parse_args(argv):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', default='results/e1')
    ap.add_argument('--trials', type=int, default=30)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--charger-pose', type=float, nargs=3, default=[8.3, -1.0, math.pi],
                    help='charger target frame in the map, as given to the simulator')
    ap.add_argument('--area', type=float, nargs=4, default=[0.5, -7.5, 6.5, -0.5],
                    metavar=('X0', 'Y0', 'X1', 'Y1'), help='start poses are sampled here')
    ap.add_argument('--keep-out', type=float, nargs='+', default=[3.0, -7.5, 1.5],
                    metavar='X Y R', help='circles to keep start poses out of')
    return ap.parse_args(remove_ros_args(args=argv if argv is not None else sys.argv)[1:])


def start_poses(args, n):
    """Reachable, repeatable start poses: a seeded uniform sample of the free area."""
    rng = random.Random(args.seed)
    x0, y0, x1, y1 = args.area
    keep_out = [tuple(args.keep_out[i:i + 3]) for i in range(0, len(args.keep_out), 3)]
    poses = []
    while len(poses) < n:
        x, y = rng.uniform(x0, x1), rng.uniform(y0, y1)
        if any(math.hypot(x - cx, y - cy) < r for cx, cy, r in keep_out):
            continue
        poses.append((x, y, rng.uniform(-math.pi, math.pi)))
    return poses


def main(argv=None):
    args = parse_args(argv)
    targets = load_targets(os.path.join(get_package_share_directory('amr_docking'),
                                        'config', 'docking_targets.yaml'))
    docked = compose(tuple(args.charger_pose), tuple(targets['charger']['robot_pose']))
    rclpy.init()
    node = Harness('e1_charger_docking')
    if not node.wait_clock():
        node.get_logger().error('No /clock: start the simulator first.')
        return 1
    log = TrialLog(args.out, 'e1_charger_docking', FIELDS)
    node.get_logger().info(f'Expected docked pose: '
                           f'({docked[0]:.3f}, {docked[1]:.3f}, {docked[2]:.3f})')
    try:
        for start in start_poses(args, args.trials):
            run_trial(node, log, start, docked)
    except KeyboardInterrupt:
        pass
    finally:
        summary = summarize(log.rows, ['dock_time', 'err_long', 'err_lat', 'err_yaw', 'err_xy'])
        summary['charging_rate'] = (sum(int(r['charging'] or 0) for r in log.rows)
                                    / len(log.rows) if log.rows else 0.0)
        log.close(summary)
        print(format_summary('E1 charger docking', summary))
        print(f"  charging confirmed in {100 * summary['charging_rate']:.1f} % of the trials")
        print(f'  log: {log.path}')
        node.destroy_node()
        rclpy.try_shutdown()
    return 0


def run_trial(node, log, start, docked):
    """Drive to the start pose, dock, measure, undock."""
    if node.latched:
        node.run_command('release_cart', timeout=180.0)
    reached, _ = node.navigate(*start, timeout=180.0)
    pose = node.odom
    distance = math.hypot(pose[0] - docked[0], pose[1] - docked[1]) if pose else float('nan')
    ok, seconds, out = node.run_command('dock_charger', timeout=420.0)
    final = node.odom or (float('nan'),) * 3
    err = pose_error(final, docked)
    charging = bool(node.battery and node.battery[2] == 1)      # POWER_SUPPLY_STATUS_CHARGING
    undock_ok = False
    if ok:
        undock_ok, _, _ = node.run_command('undock_charger', timeout=180.0)
    note = '' if ok else out.strip().splitlines()[-1][:90] if out.strip() else 'no output'
    row = log.add(start_x=f'{start[0]:.3f}', start_y=f'{start[1]:.3f}',
                  start_yaw=f'{start[2]:.3f}', start_distance=f'{distance:.3f}',
                  reached_start=int(reached), dock_time=f'{seconds:.2f}',
                  err_long=f'{err[0]:.4f}', err_lat=f'{err[1]:.4f}', err_yaw=f'{err[2]:.4f}',
                  err_xy=f'{math.hypot(err[0], err[1]):.4f}',
                  charging=int(charging), battery=f'{node.battery[0]:.3f}' if node.battery else '',
                  undock_ok=int(undock_ok), success=ok, note=note)
    n = row['trial']
    node.get_logger().info(
        f'trial {n:3d}  start ({start[0]:.2f}, {start[1]:.2f})  ok {int(ok)}'
        f'  t {seconds:.1f} s  err long {err[0]:+.3f} lat {err[1]:+.3f}'
        f' yaw {math.degrees(err[2]):+.2f} deg  charging {int(charging)}')
