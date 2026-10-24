"""E4: narrow passages with and without towing mode.

Towing mode replaces the Nav2 footprint with the convex hull of robot and cart, removes
the cart from the point cloud and caps the speed. This experiment asks what that buys:
the same corridor is driven with towing mode on and off, at several corridor widths, and
the trial records whether the robot got through and how close the cart came to the walls.

The corridor is built by the simulator (`run_sim.sh --corridor WIDTH`), so a block of
trials runs at one width and the width is swept by restarting the simulator.

The costmap inflation radius is part of the experiment, not a detail: the default 0.70 m
inflates a 1.2 m passage wall to wall, so Nav2 refuses to plan through it whatever the
footprint is. The trials therefore set the inflation radius explicitly and report it.

  ros2 run amr_experiments e4_corridor --out results/e4 --width 1.0 --repeats 3
"""

import argparse
import math
import os
import subprocess
import sys

from ament_index_python.packages import get_package_share_directory
from amr_docking.cart_towing import cart_corners, cart_pose_in_base
from amr_docking.targets import load_targets
import rclpy
from rclpy.utilities import remove_ros_args
from std_srvs.srv import SetBool

from .harness import Harness
from .trial_log import format_summary, summarize, TrialLog

FIELDS = ['width', 'towing_mode', 'pass_time', 'min_clearance', 'clearance_geom',
          'contacts', 'aborted_goals', 'beta_max', 'success', 'note']

# Parking areas on both sides of the passage. A trial always starts on the side the
# robot is already on and drives to the other one: making it come back through the
# corridor empty to fetch the cart is a second, unmeasured narrow-passage run, and that
# is what failed between trials when every pass went the same way.
# Every pass runs the same way, west to east. A towed U-turn is not available: turning
# on the spot with a cart swings it until the passive hitch hits its limit (E5), so the
# robot releases the cart on the far side and drives back through the passage empty.
# The exit pose has to leave the whole train clear of the passage, not just the robot.
# The cart trails about 0.85 m behind the hitch and is 0.84 m long, so a goal 1.1 m past
# the far wall still ends with the cart between the walls: the robot then tries to settle
# on the goal heading with the cart wedged, cannot rotate, and oscillates until the goal
# times out. That was the one failure per block at 1.2 m. 1.8 m past the wall clears it.
CART_PARK = (0.6, -4.5, 0.0)
ENTRY = (2.2, -4.5, 0.0)
EXIT = (7.3, -4.5, 0.0)


def parse_args(argv):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', default='results/e4')
    ap.add_argument('--width', type=float, required=True, help='corridor width in the sim')
    ap.add_argument('--corridor', type=float, nargs=3, default=[3.0, 5.5, -4.5],
                    metavar=('X0', 'X1', 'Y'))
    ap.add_argument('--repeats', type=int, default=3)
    ap.add_argument('--modes', nargs='+', default=['on', 'off'],
                    help='towing mode settings to compare')
    ap.add_argument('--inflation', type=float, default=0.25,
                    help='costmap inflation radius for the trials [m]')
    return ap.parse_args(remove_ros_args(args=argv if argv is not None else sys.argv)[1:])


def clearance(samples, args, tow_bar_x, half_length, half_width):
    """Smallest distance from a robot or cart corner to a corridor wall.

    The cart pose is taken from the indoor GPS when the sample has one, and derived
    from the hitch angle otherwise. Both are returned: a geometric estimate that
    disagrees with the measured pose means the hitch model, not the robot, is wrong.
    """
    x0, x1, y = args.corridor
    half = args.width / 2.0
    worst = {'measured': float('inf'), 'geometric': float('inf')}
    contacts = 0
    for sample in samples:
        rx, ry, yaw, beta = sample[1], sample[2], sample[3], sample[4]
        # Clearance only means something while the robot is actually in the passage.
        # Outside it the walls are finite, and a cart swinging wide at the mouth is in
        # open floor, not through a wall.
        if not (x0 <= rx <= x1 and abs(ry - y) < half):
            continue
        local = cart_pose_in_base(beta, tow_bar_x)
        c, s = math.cos(yaw), math.sin(yaw)
        poses = {'geometric': (rx + c * local[0] - s * local[1],
                               ry + s * local[0] + c * local[1], yaw + beta)}
        if len(sample) >= 9 and not math.isnan(sample[6]):
            poses['measured'] = (sample[6], sample[7], sample[8])
        for source, cart in poses.items():
            for cx, cy in [(rx, ry)] + cart_corners(cart, half_length, half_width):
                if x0 <= cx <= x1:
                    gap = half - abs(cy - y)
                    worst[source] = min(worst[source], gap)
                    if source == 'measured' and gap <= 0.0:
                        contacts += 1
    return ({k: (v if v < float('inf') else float('nan')) for k, v in worst.items()},
            contacts)


def main(argv=None):
    args = parse_args(argv)
    targets = load_targets(os.path.join(get_package_share_directory('amr_docking'),
                                        'config', 'docking_targets.yaml'))
    tow_bar_x = float(targets['cart']['tow_bar_in_cart'][0])
    rclpy.init()
    node = Harness('e4_corridor')
    if not node.wait_clock():
        node.get_logger().error('No /clock: start the simulator first.')
        return 1
    log = TrialLog(args.out, f'e4_corridor_{args.width:.2f}'.replace('.', 'p'), FIELDS)
    set_inflation(node, args.inflation)
    try:
        for mode in args.modes:
            run_block(node, log, args, tow_bar_x, mode == 'on')
    except KeyboardInterrupt:
        pass
    finally:
        keys = ['pass_time', 'min_clearance', 'beta_max']
        summary = {m: summarize([r for r in log.rows if int(r['towing_mode']) == (m == 'on')],
                                keys) for m in ('on', 'off')}
        summary['width'] = args.width
        summary['inflation_radius'] = args.inflation
        log.close(summary)
        for mode in ('on', 'off'):
            print(format_summary(f'E4 corridor {args.width:.2f} m, towing mode {mode}',
                                 summary[mode]))
        print(f'  log: {log.path}')
        node.destroy_node()
        rclpy.try_shutdown()
    return 0


def set_inflation(node, radius):
    """Set the costmap inflation radius for both costmaps."""
    for costmap in ('/global_costmap/global_costmap', '/local_costmap/local_costmap'):
        subprocess.run(['ros2', 'param', 'set', costmap,
                        'inflation_layer.inflation_radius', str(radius)],
                       capture_output=True, text=True, timeout=30)
    node.get_logger().info(f'costmap inflation radius set to {radius:.2f} m')
    node.spin(2.0)


def set_towing_mode(node, on):
    client = node.create_client(SetBool, '/cart/set_coupled')
    ok = False
    if client.wait_for_service(timeout_sec=5.0):
        future = client.call_async(SetBool.Request(data=on))
        rclpy.spin_until_future_complete(node, future, timeout_sec=5.0)
        ok = future.result() is not None and future.result().success
    node.destroy_client(client)
    return ok


def run_block(node, log, args, tow_bar_x, towing_mode):
    """Repeat: couple on the near side, tow through the passage, release, come back."""
    for _ in range(args.repeats):
        # Coupling is E2's subject, not this one: a hook that misses says nothing about
        # whether the passage can be driven, so try again instead of spending a trial on
        # it. Only a coupling that cannot be made at all is recorded, and then the row
        # has no pass_time, which keeps it out of the corridor statistics.
        coupled, out = False, ''
        for attempt in range(3):
            if node.latched:
                node.run_command('release_cart', timeout=180.0)
            node.reset(node.odom or ENTRY, CART_PARK, settle=2.0)   # cart only
            node.navigate(*ENTRY, timeout=240.0)
            coupled, _, out = node.run_command('dock_cart', timeout=480.0)
            if coupled:
                break
            node.get_logger().warn(f'coupling failed on attempt {attempt + 1}; retrying')
        if not coupled:
            reason = out.strip().splitlines()[-1][:90] if out.strip() else 'no output'
            log.add(width=args.width, towing_mode=int(towing_mode), success=False,
                    note=reason)
            node.get_logger().warn(f'coupling failed three times; pass skipped: {reason}')
            continue
        set_towing_mode(node, towing_mode)
        run_pass(node, log, args, tow_bar_x, towing_mode)
        node.run_command('release_cart', timeout=240.0)


def run_pass(node, log, args, tow_bar_x, towing_mode):
    """One towed run through the passage, measured from entry to exit."""
    node.navigate(*ENTRY, timeout=240.0)
    node.start_recording()
    t0 = node.time()
    ok, _ = node.navigate(*EXIT, timeout=300.0)
    samples = node.stop_recording()
    seconds = node.time() - t0
    gaps, contacts = clearance(samples, args, tow_bar_x, 0.42, 0.30)
    gap = gaps['measured'] if not math.isnan(gaps['measured']) else gaps['geometric']
    beta_max = max((abs(s[4]) for s in samples), default=float('nan'))
    row = log.add(width=args.width, towing_mode=int(towing_mode),
                  pass_time=f'{seconds:.2f}', min_clearance=f'{gap:.4f}',
                  clearance_geom=f"{gaps['geometric']:.4f}", contacts=contacts,
                  aborted_goals=int(not ok), beta_max=f'{beta_max:.4f}', success=ok)
    log.save_trajectory(row['trial'], samples)
    n = row['trial']
    node.get_logger().info(
        f'trial {n:3d}  mode {int(towing_mode)}  ok {int(ok)}  t {seconds:.1f} s'
        f'  min clearance {gap:+.3f} m  contacts {contacts}')
