"""E3: engagement detection from the hitch effort.

The robot is placed at a known offset from the tow bar, the hook is lowered, raised and
the hitch pitch effort is read before and after. The offset is the independent variable:
small offsets catch the bar, large ones miss, and offsets around the hook mouth give the
partial engagements that a threshold has to reject.

Only the simulator can place the robot this precisely, so this is the one experiment that
cannot be repeated on hardware the same way; on the robot the same quantity is measured
by hand-placing the cart. What transfers is the shape of the two distributions and the
threshold that separates them (in Nm here, in Dynamixel current on the robot).

  ros2 run amr_experiments e3_engagement -- --out results/e3 --repeats 5
"""

import argparse
import math
import os
import sys

from ament_index_python.packages import get_package_share_directory
from amr_docking.targets import compose, load_targets
import rclpy
from rclpy.utilities import remove_ros_args
from sensor_msgs.msg import JointState

from .harness import Harness
from .trial_log import describe, format_summary, numbers, TrialLog

HOOK_DOWN = -0.2        # rad, same defaults as the amr_docking commands
HOOK_UP = 0.2
FIELDS = ['offset_long', 'offset_lat', 'cart_mass', 'effort_free', 'effort_loaded',
          'latched', 'detected', 'correct', 'success']


def parse_args(argv):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', default='results/e3')
    ap.add_argument('--repeats', type=int, default=5)
    ap.add_argument('--threshold', type=float, default=0.5, help='engage_effort under test')
    ap.add_argument('--cart-mass', type=float, default=22.0, help='mass set in the simulator')
    ap.add_argument('--cart-pose', type=float, nargs=3, default=[3.0, -7.5, math.pi / 2])
    ap.add_argument('--offsets', type=float, nargs='+',
                    default=[0.0, 0.01, 0.02, 0.025, 0.03, 0.035, 0.04, 0.06, 0.10],
                    help='lateral hook-to-bar offsets [m]')
    ap.add_argument('--longitudinal', type=float, nargs='+', default=[0.0, 0.02, -0.02],
                    help='longitudinal offsets [m]')
    return ap.parse_args(remove_ros_args(args=argv if argv is not None else sys.argv)[1:])


class Engagement(Harness):

    def __init__(self, targets):
        super().__init__('e3_engagement')
        self.targets = targets
        self.hitch = self.create_publisher(JointState, '/amr_hitch/command', 10)

    def docked_pose(self, cart_pose):
        """Robot pose that puts the hook on the tow bar, from the target geometry."""
        cfg = self.targets['cart']
        target = compose(cart_pose, tuple(cfg['target_in_cart']))
        return compose(target, tuple(cfg['robot_pose']))

    def set_hook(self, pitch, settle=1.5):
        self.hitch.publish(JointState(name=['hitch_yaw_joint', 'hitch_pitch_joint'],
                                      position=[0.0, pitch]))
        self.spin(settle)

    def mean_effort(self, seconds=0.6):
        readings = []
        end = self.time() + seconds
        while self.time() < end:
            rclpy.spin_once(self, timeout_sec=0.02)
            readings.append(self.pitch_effort())
        return sum(readings) / len(readings) if readings else 0.0

    def trial(self, cart_pose, offset_long, offset_lat):
        """One place-lower-raise-read cycle. Returns the measured row."""
        docked = self.docked_pose(cart_pose)
        start = compose(docked, (offset_long, offset_lat, 0.0))
        self.set_hook(HOOK_DOWN, settle=0.2)
        self.reset(start, cart_pose, settle=1.5)
        self.set_hook(HOOK_DOWN)
        free = self.mean_effort()
        self.set_hook(HOOK_UP)
        loaded = self.mean_effort()
        latched = self.latched
        self.set_hook(HOOK_DOWN, settle=1.0)
        return free, loaded, latched


def main(argv=None):
    args = parse_args(argv)
    targets = load_targets(os.path.join(get_package_share_directory('amr_docking'),
                                        'config', 'docking_targets.yaml'))
    rclpy.init()
    node = Engagement(targets)
    if not node.wait_clock():
        node.get_logger().error('No /clock: start the simulator first.')
        return 1
    log = TrialLog(args.out, 'e3_engagement', FIELDS)
    cart_pose = tuple(args.cart_pose)
    try:
        for offset_long in args.longitudinal:
            for offset_lat in args.offsets:
                for _ in range(args.repeats):
                    free, loaded, latched = node.trial(cart_pose, offset_long, offset_lat)
                    detected = loaded >= args.threshold
                    row = log.add(offset_long=f'{offset_long:.3f}',
                                  offset_lat=f'{offset_lat:.3f}',
                                  cart_mass=args.cart_mass,
                                  effort_free=f'{free:.4f}', effort_loaded=f'{loaded:.4f}',
                                  latched=int(latched), detected=int(detected),
                                  correct=int(detected == latched), success=True)
                    n = row['trial']
                    node.get_logger().info(
                        f'trial {n:3d}  long {offset_long:+.3f}  lat {offset_lat:.3f}'
                        f'  free {free:.3f}  loaded {loaded:.3f}'
                        f'  latched {int(latched)}  detected {int(detected)}')
    except KeyboardInterrupt:
        pass
    finally:
        report(node, log, args)
        node.destroy_node()
        rclpy.try_shutdown()
    return 0


def report(node, log, args):
    rows = log.rows
    hooked = [r for r in rows if r['latched'] == 1]
    missed = [r for r in rows if r['latched'] == 0]
    correct = sum(int(r['correct']) for r in rows)
    summary = {
        'threshold': args.threshold,
        'cart_mass': args.cart_mass,
        'trials': len(rows),
        'hooked': len(hooked),
        'missed': len(missed),
        'accuracy': correct / len(rows) if rows else 0.0,
        'effort_hooked': describe(numbers(hooked, 'effort_loaded')),
        'effort_missed': describe(numbers(missed, 'effort_loaded')),
        'effort_free': describe(numbers(rows, 'effort_free')),
    }
    gap = (summary['effort_hooked'].get('min', 0.0) - summary['effort_missed'].get('max', 0.0)
           if hooked and missed else 0.0)
    summary['separation'] = gap
    log.close(summary)
    print(format_summary('E3 engagement detection', {
        'trials': len(rows), 'successes': correct,
        'success_rate': summary['accuracy'],
        'effort_loaded_hooked': summary['effort_hooked'],
        'effort_loaded_missed': summary['effort_missed'],
    }))
    print(f'  separation between the two distributions: {gap:+.3f} Nm')
    print(f'  log: {log.path}')
