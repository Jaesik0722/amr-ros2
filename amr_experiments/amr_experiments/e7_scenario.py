"""E7: the whole mission in one simulation run.

E1-E6 each isolate one capability and repeat it until its numbers are trustworthy. This
runner does the opposite: it drives the complete mission once, end to end, in a single
world, so that what every experiment measured separately is seen happening in sequence
and in the state the previous stage left behind.

  fetch the cart (E2) -> confirm the coupling from the hitch effort (E3) ->
  tow through the narrow passage in towing mode (E4) -> run the delivery leg and
  measure off-tracking (E5) -> release (E2) -> dock to the wireless charger and
  confirm charging (E1) -> undock and come home empty through the same passage (E6)

The passage is driven twice, towed on the way out and empty on the way back, which is
the comparison E4 makes across blocks of trials, here inside one run.

The point of the scenario is integration, not statistics: one run of each stage says
nothing about repeatability, and the per-stage numbers it prints are there to be
recognised against the campaign tables, not to replace them. The simulator has to be
started with the passage in place, which is also what puts the charger behind it:

  AMR_CART_WHEELS=trailer ./amr_experiments/tools/run_experiment.sh e7

  ros2 run amr_experiments e7_scenario --out results/e7 --width 1.00

Stage order and the geometry come from the experiments the stages are taken from, so the
poses here are the same constants: a number that moved would make the scenario
incomparable with the campaign it is meant to illustrate.
"""

import argparse
import math
import os
import re
import sys

from ament_index_python.packages import get_package_share_directory
from amr_docking.targets import compose, load_targets
import rclpy
from rclpy.utilities import remove_ros_args

from .corridor_clearance import average_cart, measure
from .e4_corridor import set_inflation, set_towing_mode
from .e5_towing import analyse
from .harness import Harness, pose_error
from .trial_log import TrialLog

EFFORT_RE = re.compile(r'effort: ([0-9.]+) before, ([0-9.]+) after')
FIELDS = ['stage', 'title', 'ok', 'seconds', 'detail', 'note']

# E4's parking and passage poses, and E5's delivery leg, unchanged.
CART_PARK = (0.6, -4.5, 0.0)
ENTRY = (2.2, -4.5, 0.0)
EXIT = (7.3, -4.5, 0.0)
# The delivery run leaves the passage and sweeps north through two legs, so the towed
# route has corners in it and the off-tracking E5 measures is visible here too.
#
# It used to run south-east to (4.5, -7.2) instead. That works, but it leaves the robot
# in the corner between the wing wall and the south edge of the floor with the cart it
# just dropped between it and the charger, and the charger dock then has to thread back
# out past both. It did, headless; under the load of the video capture it did not, and
# the mission stopped at stage 6 with the robot 6 m from the pad. Ending in open floor
# north of the passage costs nothing and removes the whole situation.
DELIVERY_ROUTE = [(7.0, -2.5), (5.0, -2.0)]
HOME = (1.0, -1.0, 0.0)
CHARGE_SECONDS = 20.0


def parse_args(argv):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', default='results/e7')
    ap.add_argument('--width', type=float, default=1.00,
                    help='corridor width the simulator was started with')
    ap.add_argument('--corridor', type=float, nargs=3, default=[3.0, 5.5, -4.5],
                    metavar=('X0', 'X1', 'Y'))
    ap.add_argument('--charger-pose', type=float, nargs=3, default=[8.3, -1.0, math.pi])
    ap.add_argument('--inflation', type=float, default=0.25,
                    help='costmap inflation radius, as in E4')
    ap.add_argument('--charge-seconds', type=float, default=CHARGE_SECONDS)
    return ap.parse_args(remove_ros_args(args=argv if argv is not None else sys.argv)[1:])


class Mission:
    """Runs the stages in order and records one row per stage."""

    def __init__(self, node, log, args, targets):
        self.node = node
        self.log = log
        self.args = args
        self.targets = targets
        self.tow_bar_x = float(targets['cart']['tow_bar_in_cart'][0])
        self.effort_free = float('nan')
        self.effort_loaded = float('nan')
        self.note = ''          # a command's last line, carried into the next row

    def record(self, stage, title, ok, seconds, detail):
        self.log.add(stage=stage, title=title, ok=int(bool(ok)),
                     seconds=f'{seconds:.2f}', detail=detail, note=self.note)
        self.note = ''
        # The caption in the video is this line, so it has to read as a sentence about
        # what just happened rather than as a row of fields.
        self.node.get_logger().info(
            f'stage {stage}  {title}  {"ok" if ok else "FAILED"}  {seconds:.0f} s  {detail}')
        return ok

    # ---- stages --------------------------------------------------------------
    def fetch_cart(self):
        """E2: drive to the staging pose and reverse-dock onto the tow bar."""
        cfg = self.targets['cart']
        expected = compose(compose(CART_PARK, tuple(cfg['target_in_cart'])),
                           tuple(cfg['robot_pose']))
        self.node.reset(self.node.odom or ENTRY, CART_PARK, settle=2.0)   # cart only
        self.node.navigate(*ENTRY, timeout=240.0)
        ok, seconds, out = self.node.run_command('dock_cart', timeout=480.0)
        err = pose_error(self.node.odom or (float('nan'),) * 3, expected)
        match = EFFORT_RE.search(out)
        if match:
            self.effort_free = float(match.group(1))
            self.effort_loaded = float(match.group(2))
        detail = (f'docked {math.hypot(err[0], err[1]) * 1000:.0f} mm from the tow bar '
                  f'(long {err[0] * 1000:+.0f}, lat {err[1] * 1000:+.0f} mm, '
                  f'yaw {math.degrees(err[2]):+.1f} deg)')
        return self.record('1/7', 'E2  reverse docking to the cart', ok, seconds, detail)

    def confirm_coupling(self):
        """E3: the hitch effort before and after the lift is what says the bar is in."""
        threshold = 0.5
        ok = (self.effort_loaded >= threshold) and self.node.latched
        detail = (f'hitch effort {self.effort_free:.3f} -> {self.effort_loaded:.3f} Nm '
                  f'(threshold {threshold:.2f}), latched {int(self.node.latched)}')
        return self.record('2/7', 'E3  coupling confirmed from the hitch effort',
                           ok, 0.0, detail)

    def narrow_passage(self):
        """E4: the towed pass, with towing mode on and the measured clearance."""
        set_towing_mode(self.node, True)
        self.node.navigate(*ENTRY, timeout=240.0)
        self.node.start_recording()
        t0 = self.node.time()
        ok, _ = self.node.navigate(*EXIT, timeout=300.0)
        samples = self.node.stop_recording()
        seconds = self.node.time() - t0
        # The same correction the campaign uses: the raw indoor-GPS clearance is an
        # extreme value of that sensor's own noise (corridor_clearance).
        track = average_cart([(x[0], x[1], x[2], (x[6], x[7], x[8])) for x in samples
                              if not math.isnan(x[6])])
        gap, _, touching = measure(track, self.args.width, tuple(self.args.corridor))
        beta_max = max((abs(s[4]) for s in samples), default=float('nan'))
        self.log.save_trajectory(len(self.log.rows) + 1, samples)
        detail = (f'towed through the {self.args.width:.2f} m passage, clearance '
                  f'{gap * 1000:+.0f} mm, hitch angle up to {math.degrees(beta_max):.0f} deg, '
                  f'{touching:.1f} s against a wall')
        return self.record('3/7', 'E4  narrow passage in towing mode', ok, seconds, detail)

    def delivery_leg(self):
        """E5: the long towed route, measured for off-tracking and hitch angle."""
        # E5's rule for a towed goal: face the way the robot arrives. A goal heading that
        # differs from the direction of travel asks the robot to turn on the spot, and an
        # off-axle hitch then drags the cart until the passive yaw joint hits its limit.
        previous = (self.node.odom or EXIT)[:2]
        self.node.start_recording()
        t0 = self.node.time()
        ok = True
        for point in DELIVERY_ROUTE:
            heading = math.atan2(point[1] - previous[1], point[0] - previous[0])
            reached, _ = self.node.navigate(point[0], point[1], heading, timeout=300.0)
            ok = ok and reached
            previous = point
        samples = self.node.stop_recording()
        seconds = self.node.time() - t0
        stats = analyse(samples, self.tow_bar_x, True)
        final = self.node.odom or (float('nan'),) * 3
        goal_err = math.dist(final[:2], DELIVERY_ROUTE[-1])
        self.log.save_trajectory(len(self.log.rows) + 1, samples)
        detail = (f'off-tracking max {stats.get("offtrack_max", float("nan")) * 1000:.0f} mm, '
                  f'rms {stats.get("offtrack_rms", float("nan")) * 1000:.0f} mm, '
                  f'hitch angle up to {math.degrees(stats.get("beta_max", float("nan"))):.0f} '
                  f'deg, stopped {goal_err * 1000:.0f} mm from the goal')
        return self.record('4/7', 'E5  delivery leg with the cart', ok, seconds, detail)

    def release(self):
        """E2: unhook, confirm the load has come off and the latch has opened.

        One retry. A release that reports success and leaves the cart on the hook
        happened in 2 of 30 trials of the four-caster E2 campaign and once here with the
        trailer, so it is a real fault at the percent level rather than a cart property,
        and a system that runs unattended has to try again before it stops.
        """
        ok, seconds, out = self.node.run_command('release_cart', timeout=240.0)
        self.node.spin(2.0)
        if self.node.latched or self.node.pitch_effort() > 0.5:
            self.node.get_logger().warn(
                f'the cart is still on the hook ({self.node.pitch_effort():.2f} Nm, '
                f'latched {int(self.node.latched)}); releasing again')
            ok, more, out = self.node.run_command('release_cart', timeout=240.0)
            seconds += more
        deadline = self.node.time() + 8.0
        while self.node.latched and self.node.time() < deadline:
            self.node.spin(0.25)
        effort = self.node.pitch_effort()
        # The command's own last word goes in the CSV, not in the caption: a release that
        # reports success and leaves the cart on the hook is the interesting one (E2), and
        # the line is too long to read under the picture.
        note = (out.strip().splitlines() or ['no output'])[-1][:90]
        self.note = note
        detail = (f'hitch effort back to {effort:.3f} Nm, '
                  f'latched {int(self.node.latched)}')
        return self.record('5/7', 'E2  cart released at the delivery pose',
                           ok and not self.node.latched, seconds, detail)

    def charge(self):
        """E1: forward dock onto the wireless charger and confirm the current."""
        docked = compose(tuple(self.args.charger_pose),
                         tuple(self.targets['charger']['robot_pose']))
        # One retry. The dock has to come in past the cart that was just dropped, and a
        # real robot that misses its charger tries again before calling for help.
        ok, seconds, _ = self.node.run_command('dock_charger', timeout=420.0)
        if not ok:
            self.node.get_logger().warn('charger dock failed; trying once more')
            again, more, _ = self.node.run_command('dock_charger', timeout=420.0)
            ok, seconds = again, seconds + more
        err = pose_error(self.node.odom or (float('nan'),) * 3, docked)
        if ok:
            self.node.spin(self.args.charge_seconds)
        charging = bool(self.node.battery and self.node.battery[2] == 1)
        current = self.node.battery[1] if self.node.battery else float('nan')
        detail = (f'docked {math.hypot(err[0], err[1]) * 1000:.0f} mm from the pad '
                  f'(long {err[0] * 1000:+.0f}, lat {err[1] * 1000:+.0f} mm), '
                  f'charging {int(charging)} at {current:+.2f} A')
        return self.record('6/7', 'E1  wireless charger docking', ok and charging,
                           seconds, detail)

    def go_home(self):
        """E6: undock and come back through the same passage with no cart."""
        undock_ok, seconds, _ = self.node.run_command('undock_charger', timeout=180.0)
        self.node.start_recording()
        t0 = self.node.time()
        # By way of the west mouth of the passage. Sent straight home instead, the robot
        # goes round the north end of the wing wall, which is the shorter path from the
        # charger and leaves the empty return - the half of E4 this clip is here to show
        # next to the towed one - never driven.
        ok, _ = self.node.navigate(*ENTRY, timeout=300.0)
        reached, _ = self.node.navigate(*HOME, timeout=300.0)
        ok = ok and reached
        samples = self.node.stop_recording()
        seconds += self.node.time() - t0
        self.log.save_trajectory(len(self.log.rows) + 1, samples)
        # Only the samples actually between the walls. Testing the x range alone also
        # catches the open floor north of the passage, which the robot crosses on its
        # way back from the charger: that put 5.8 m of "offset from the centreline" in
        # the caption of the first take.
        x0, x1, y = self.args.corridor
        half = self.args.width / 2.0
        inside = [abs(s[2] - y) for s in samples
                  if x0 <= s[1] <= x1 and abs(s[2] - y) < half]
        if inside:
            detail = (f'undocked {int(undock_ok)}, home through the same passage with no '
                      f'cart, centre within {max(inside) * 1000:.0f} mm of the centreline')
        else:
            detail = (f'undocked {int(undock_ok)}, home without entering the passage')
        return self.record('7/7', 'E6  undock and return home empty',
                           undock_ok and ok, seconds, detail)


def main(argv=None):
    args = parse_args(argv)
    targets = load_targets(os.path.join(get_package_share_directory('amr_docking'),
                                        'config', 'docking_targets.yaml'))
    rclpy.init()
    node = Harness('e7_scenario')
    if not node.wait_clock():
        node.get_logger().error('No /clock: start the simulator first.')
        return 1
    log = TrialLog(args.out, 'e7_scenario', FIELDS)
    set_inflation(node, args.inflation)
    mission = Mission(node, log, args, targets)
    t0 = node.time()
    try:
        # Each stage needs the one before it: a cart that was never hooked cannot be
        # towed through the passage. A stage that fails therefore ends the mission, and
        # the rows written so far say how far it got.
        for stage in (mission.fetch_cart, mission.confirm_coupling, mission.narrow_passage,
                      mission.delivery_leg, mission.release, mission.charge,
                      mission.go_home):
            if not stage():
                node.get_logger().error('stage failed; the mission stops here')
                break
    except KeyboardInterrupt:
        pass
    finally:
        report(log, node.time() - t0, args)
        node.destroy_node()
        rclpy.try_shutdown()
    return 0


def report(log, seconds, args):
    rows = log.rows
    done = sum(int(r['ok']) for r in rows)
    meta = {'stages': len(rows), 'stages_ok': done, 'mission_seconds': seconds,
            'corridor_width': args.width, 'inflation_radius': args.inflation,
            'complete': done == 7}
    log.close(meta)
    print(f'--- E7 scenario: {done}/7 stages, {seconds / 60.0:.1f} min ---')
    for row in rows:
        print(f"  {row['stage']}  {'ok   ' if int(row['ok']) else 'FAIL '}"
              f"{row['title']:<44} {row['detail']}")
    print(f'  log: {log.path}')


if __name__ == '__main__':
    main()
