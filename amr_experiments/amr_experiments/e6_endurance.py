"""E6: unattended operation over many mission cycles.

One cycle is the whole mission the system is built for:

  dock to the charger -> charge -> undock -> navigate to the cart -> couple ->
  tow to the delivery pose -> release -> return

The trial never resets the world between cycles, so faults accumulate the way they do on
a real shift: a stale costmap, a cart left at an angle, a battery that keeps draining.
What the experiment reports is the cycle success rate, the mean time between
interventions and what each failure was, which is the evidence a systems paper needs
that the thing runs on its own.

An intervention is counted whenever a cycle cannot continue and the runner has to put
the world back (release a stuck cart, teleport the robot clear).

  ros2 run amr_experiments e6_endurance --out results/e6 --cycles 30
"""

import argparse
import math
import os
import sys

from ament_index_python.packages import get_package_share_directory
from amr_docking.targets import load_targets
import rclpy
from rclpy.utilities import remove_ros_args

from .harness import Harness
from .trial_log import format_summary, summarize, TrialLog

FIELDS = ['cycle_time', 'dock_charger', 'charge_pct', 'undock', 'dock_cart', 'tow', 'release',
          'return_ok', 'intervention', 'failure_stage', 'battery_start', 'battery_end',
          'success']

CART_POSE = (3.0, -7.5, math.pi / 2)
# The delivery heading is the bearing of the leg that reaches it: a towed robot that has
# to turn on the spot at the goal drags the cart sideways until the hitch hits its limit.
DELIVERY = (7.0, -2.5, math.atan2(-2.5 - CART_POSE[1], 7.0 - CART_POSE[0]))
HOME = (1.0, -1.0, 0.0)


def parse_args(argv):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', default='results/e6')
    ap.add_argument('--cycles', type=int, default=30)
    ap.add_argument('--charge-seconds', type=float, default=20.0,
                    help='time spent on the charger each cycle')
    return ap.parse_args(remove_ros_args(args=argv if argv is not None else sys.argv)[1:])


def main(argv=None):
    args = parse_args(argv)
    load_targets(os.path.join(get_package_share_directory('amr_docking'),
                              'config', 'docking_targets.yaml'))
    rclpy.init()
    node = Harness('e6_endurance')
    if not node.wait_clock():
        node.get_logger().error('No /clock: start the simulator first.')
        return 1
    node.spin(3.0)      # the first /battery_state, so cycle 1 has a start charge
    log = TrialLog(args.out, 'e6_endurance', FIELDS)
    try:
        for _ in range(args.cycles):
            run_cycle(node, log, args)
    except KeyboardInterrupt:
        pass
    finally:
        report(log)
        node.destroy_node()
        rclpy.try_shutdown()
    return 0


def run_cycle(node, log, args):
    """One full mission. Returns nothing; the row says where it stopped."""
    stages = {}
    t0 = node.time()
    battery_start = node.battery[0] if node.battery else float('nan')
    failure = ''
    intervention = 0

    stages['dock_charger'], _, _ = node.run_command('dock_charger', timeout=420.0)
    if stages['dock_charger']:
        node.spin(args.charge_seconds)
    else:
        failure = 'dock_charger'
    charge_pct = node.battery[0] if node.battery else float('nan')

    stages['undock'] = False
    if stages['dock_charger']:
        stages['undock'], _, _ = node.run_command('undock_charger', timeout=180.0)
        if not stages['undock']:
            failure = failure or 'undock'

    stages['dock_cart'] = False
    if stages['undock']:
        stages['dock_cart'], _, _ = node.run_command('dock_cart', timeout=480.0)
        if not stages['dock_cart']:
            failure = failure or 'dock_cart'

    stages['tow'] = False
    if stages['dock_cart']:
        stages['tow'], _ = node.navigate(*DELIVERY, timeout=300.0)
        if not stages['tow']:
            failure = failure or 'tow'

    stages['release'] = False
    if node.latched:
        stages['release'], _, _ = node.run_command('release_cart', timeout=240.0)
        if not stages['release']:
            failure = failure or 'release'

    stages['return_ok'], _ = node.navigate(*HOME, timeout=300.0)
    if not stages['return_ok']:
        failure = failure or 'return'

    if failure:
        intervention = 1
        recover(node)

    row = log.add(cycle_time=f'{node.time() - t0:.2f}',
                  dock_charger=int(stages['dock_charger']),
                  charge_pct=f'{charge_pct:.3f}', undock=int(stages['undock']),
                  dock_cart=int(stages['dock_cart']), tow=int(stages['tow']),
                  release=int(stages['release']), return_ok=int(stages['return_ok']),
                  intervention=intervention, failure_stage=failure,
                  battery_start=f'{battery_start:.3f}',
                  battery_end=f'{node.battery[0]:.3f}' if node.battery else '',
                  success=not failure)
    n = row['trial']
    battery_end = node.battery[0] if node.battery else float('nan')
    stage = failure or 'none'
    node.get_logger().info(
        f'cycle {n:3d}  success {int(not failure)}  t {node.time() - t0:.0f} s'
        f'  battery {battery_start:.2f} -> {battery_end:.2f}  failure {stage}')


def recover(node):
    """Put the world back so the next cycle can start.

    The cart is teleported back to its parking pose, but the robot drives home: a
    teleported robot breaks the SLAM pose estimate and every later cycle with it.
    """
    if node.latched:
        node.run_command('release_cart', timeout=240.0)
    node.reset(node.odom or HOME, CART_POSE, settle=2.0)
    node.navigate(*HOME, timeout=300.0)


def report(log):
    rows = log.rows
    summary = summarize(rows, ['cycle_time', 'charge_pct'])
    interventions = sum(int(r['intervention']) for r in rows)
    total_time = sum(float(r['cycle_time']) for r in rows if r['cycle_time'])
    summary['interventions'] = interventions
    summary['total_hours'] = total_time / 3600.0
    summary['mtbi_minutes'] = (total_time / 60.0 / interventions) if interventions else None
    stages = {}
    for key in ('dock_charger', 'undock', 'dock_cart', 'tow', 'release', 'return_ok'):
        attempted = [r for r in rows if r[key] != '']
        stages[key] = (sum(int(r[key]) for r in attempted), len(attempted))
    summary['stages'] = stages
    failures = {}
    for row in rows:
        if row['failure_stage']:
            failures[row['failure_stage']] = failures.get(row['failure_stage'], 0) + 1
    summary['failures'] = failures
    log.close(summary)
    print(format_summary('E6 endurance', summary))
    hours, mtbi = summary['total_hours'], summary['mtbi_minutes']
    # With no intervention the MTBI is censored by the length of the run, so report the
    # bound the campaign actually supports instead of an infinity nobody can cite.
    mtbi_text = f'MTBI {mtbi:.1f} min' if mtbi else f'MTBI > {hours * 60.0:.0f} min (censored)'
    print(f'  total {hours:.2f} h, {interventions} interventions, {mtbi_text}')
    for key, (ok, total) in stages.items():
        print(f'  {key:<14} {ok}/{total}')
    if failures:
        print(f'  failures: {failures}')
    print(f'  log: {log.path}')
