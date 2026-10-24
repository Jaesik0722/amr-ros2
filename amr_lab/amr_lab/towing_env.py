"""Learning task: drive the towed cart to a commanded pose, with an active hitch.

What the campaign in ``amr_experiments`` measured is the passive case: the hitch yaw
servo is limp and the cart trails. Every towing run ended up against the +-90 deg joint
limit, one in ten jackknifed, and the robot could not back the cart at all. The hitch
yaw servo is the one actuator on this robot that nothing has used yet, and this task is
about whether a policy can use it - including to reverse, which the passive hitch cannot
do at any speed.

The task
    The cart starts coupled (one articulation, see ``asset.py``) and a pose is commanded
    for the *cart*, not the robot. Reaching it may need forward towing, a turn on the
    spot, or backing the cart up; nothing tells the policy which, and goals behind the
    cart are sampled on purpose.

    ``cfg.bay`` replaces the pose on an empty floor with the job the application has:
    back the cart into a slot between two racks. The slot is 0.80 m of clear span for a
    0.56 m deck, so there are 120 mm a side, and the cart can only be in it nose-out,
    which means every episode ends in a reverse. It is the harder statement of the same
    question - with a free hitch the cart's heading is not actuated, and in a slot that
    is not a matter of degree - and it moves the precision the task demands out of the
    author's tolerance and into the geometry. The free-space goal stays the default so
    that the recorded campaign and its checkpoints keep meaning what they meant.

What the actor sees
    Only what the real robot can measure, at the rate it measures it: forward speed and
    yaw rate from wheel odometry at 20 Hz, the hitch servo's own position, velocity and
    current, and the goal expressed against the cart pose *predicted* from the localised
    robot pose and the measured hitch angle. For a single-axle trailer that prediction is
    exact. For the four-caster cart it is not - nothing holds the cart's heading, which
    is what the two-cart comparison found - so the policy has to work from a cart pose it
    cannot fully observe. ``cfg.true_cart_obs`` turns that into the ablation.

    Every channel carries the device's noise, its quantisation and a calibration error
    that lasts the episode, and commands arrive a delay late. See the sim-to-real block
    in the cfg: the point of this task is a policy that transfers, and an observation the
    robot cannot reproduce is worse than no observation at all.

What the critic sees
    The same, plus the true cart pose and velocity and the randomised parameters. An
    asymmetric critic is what lets the actor learn to infer the payload from the hitch
    instead of being told it.

Payload
    ``payload_mass_range`` is the headline sweep rather than a detail. A 10.6 Nm hitch
    servo cannot drag a loaded cart round against its casters: the lateral force needed
    is mu*m*g on a lever of about 0.47 m, which is tens of Nm at 23 kg and worse under
    load. So the hitch can only steer while the cart is already rolling, and how much
    payload that still holds for is a number this task produces. ``hitch_effort_range``
    randomises the servo limit alongside it, so one policy covers the hitch that is on
    the robot and a larger one, and the two sweeps can be read against each other.
"""

from __future__ import annotations

import math

import isaaclab.sim as sim_utils
import torch
from isaaclab.actuators import DCMotorCfg, ImplicitActuatorCfg  # noqa: I100
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import euler_xyz_from_quat, wrap_to_pi
from pxr import Gf, UsdGeom  # noqa: I100

from . import asset

# Robot constants that the task needs; the rest live in asset.py.
WHEEL_RADIUS = asset.WHEEL_RADIUS
WHEEL_SEPARATION = 2 * asset.WHEEL_Y
# base_footprint -> hook tip -> cart centre, with the hitch straight. This is the lever
# the kinematic cart-pose prediction uses.
HOOK_FROM_YAW = asset.PITCH_IN_YAW[0] + asset.TIP_IN_PITCH[0]
YAW_AXIS_X = asset.YAW_IN_BASE[0]
CART_FROM_HOOK = asset.TOW_BAR_IN_CART[0]
# Cart centre from the hitch yaw axis, along the cart's own axis (negative:
# the cart is behind the robot). This is the lever the hitch works through.
CART_LEVER = HOOK_FROM_YAW - CART_FROM_HOOK


@configclass
class TowingEnvCfg(DirectRLEnvCfg):
    """Everything the task needs; the CLI in train.py overrides the sweep ranges."""

    # 200 Hz physics as in the ROS simulator. The control rate is the real robot's:
    # amr_base drives the MD driver's velocity loop and publishes odometry at 20 Hz
    # (`update_rate` in base_node.py), so a policy stepped faster than that would issue
    # wheel commands the hardware never sees and read odometry it would not have. The
    # hitch node runs at 50 Hz but holds its last target, so 20 Hz costs it nothing.
    decimation = 10
    episode_length_s = 15.0
    action_space = 3
    observation_space = 0           # filled in by __post_init__
    state_space = 0
    # m/s, rad/s, rad/s. The hitch action is a *rate* on the commanded angle, not the
    # angle itself. Commanding the angle directly lets the policy ask for 1.4 rad in one
    # control step; with Kp = 50 the servo slams the joint past its +-90 deg limit and
    # the episode ends as a jackknife 0.3 s in, which teaches the policy that the hitch
    # is to be left alone - the opposite of the point. A rate also matches the servo,
    # which has a velocity limit of its own (3.14 rad/s in the URDF).
    action_scale = (0.5, 1.5, 2.0)
    beta_command_limit = 1.2        # rad, how far the commanded angle may go

    cart_wheels = 'trailer'
    active_hitch = True
    # Third mechanical condition between the driven servo and the limp one: the servo
    # holds beta at zero and the policy does not command it. It answers the question a
    # driven-versus-limp pair cannot - whether the hitch has to be *modulated*, or
    # whether simply denying it the freedom to fold is enough. Action space is two, as
    # for the passive case, so the learner and its budget are unchanged.
    lock_hitch = False
    # A fourth condition, and the one the locked case above is often mistaken for. The
    # locked case is what a *zero reference on this servo* achieves: the load drags the
    # joint off target, to a peak of 33 deg on average, so what it measures is the
    # authority of a 10.6 N.m servo asked to hold still. A rigid lock is a mechanical
    # claim instead - a welded drawbar, which no load moves - and it is a different
    # system. The joint is pinned by the solver rather than by a stiffer gain, because a
    # gain high enough to be rigid at 200 Hz is also high enough to be unstable.
    rigid_lock = False
    rigid_lock_play = 0.002         # rad, the span the pinned limits leave the solver
    # Scenery for a recording: Isaac's Simple_Warehouse, the same environment the ROS
    # simulator runs in, so a clip from here cuts against the mission recordings. It is
    # off for training -- it is a few hundred thousand triangles per environment and the
    # task never sees it, since no observation is visual.
    backdrop = False
    # Integrator and solver settings, exposed so a trained checkpoint can be replayed
    # under different numerics. A result that moves when the physics step or the solver
    # iteration count moves is a property of the simulation setup rather than of the
    # machine being simulated, and the two have to be told apart.
    # The hitch servo as a torque source that does not care how fast it is turning is
    # the generous reading of a Dynamixel. 10.6 N.m is its *stall* torque: the value at
    # zero speed and maximum current, which is neither a continuous rating nor available
    # once the joint is moving. ``hitch_dc`` replaces the flat limit with the linear
    # torque-speed curve a DC motor actually has,
    #     tau_max(w) = tau_stall * (1 - |w| / w_noload) ,
    # with the no-load speed the URDF's joint velocity limit. At the 2.0 rad/s the policy
    # may command, that leaves about a third of stall. Thermal derating is still not
    # modelled.
    hitch_dc = False
    hitch_noload_speed = 3.14          # rad/s, the URDF joint velocity limit
    solver_position_iterations = 8
    solver_velocity_iterations = 1
    # Swivel angle the four casters start an episode at. Zero for every caster is the
    # pose the asset is authored in, and a cart the robot walks up to in a factory is not
    # in it; leaving it fixed lets a policy rely on a starting condition that will not
    # hold. The range is half-width in rad about straight ahead.
    caster_init_range = (0.0, 0.0)
    history = 4                     # control frames of observation history
    true_cart_obs = False           # ablation: give the actor the true cart pose

    # ---- sim to real ----------------------------------------------------------------
    # Two channels of the first version of this observation were not measurable on the
    # robot: the body-frame *lateral* velocity, which differential-drive odometry defines
    # as zero and no sensor here reports (it is the slip the hitch side-force causes -
    # exactly the thing a policy would learn to lean on), and the tow bar angle, which
    # turns in an unsensored bearing inside the hook. `observable_only` drops both;
    # setting it False restores them, which makes the pair an ablation of what a policy
    # gives up by being honest about its sensors.
    observable_only = True
    # Noise, quantisation and a per-episode calibration error on everything that is
    # left, sized from the devices in amr_base/amr_hitch rather than guessed. Off
    # reproduces a noiseless run.
    sensor_noise = True
    # Actuation delay in control steps of 50 ms: cmd_vel through the 19200 baud serial
    # link into the driver's own loop. Zero to two spans the round trip and a dropped
    # frame.
    latency_steps_range = (0, 2)
    odom_v_noise = 0.01                     # m/s, jitter on the reported rpm
    odom_w_noise = 0.02                     # rad/s
    # Wheel radius against a loaded tyre, and the track width: both are odometry scale
    # errors, constant within a run, and why real odometry over-reads under load.
    odom_v_scale_range = (0.97, 1.03)
    odom_w_scale_range = (0.98, 1.02)
    # The driver reports integer motor rpm, so odometry arrives in steps of
    # 2*pi*r/60/gear = 2.4 mm/s, and twice that across the track in yaw rate.
    odom_v_quantum = 0.0024                 # m/s
    odom_w_quantum = 0.0129                 # rad/s
    # XM540: 4096 ticks/rev, and `zero_ticks` in hitch_node.py is a hand calibration.
    # A couple of degrees of zero error is the honest figure and it matters more than
    # the noise, because the cart pose is *predicted* from this angle: the error walks
    # straight into where the policy thinks the cart is, as it will on the robot.
    beta_noise = 0.003                      # rad
    beta_offset_range = (-0.035, 0.035)     # rad, +-2 deg
    beta_dot_noise = 0.02                   # rad/s
    beta_quantum = 2 * math.pi / 4096       # rad
    hitch_torque_noise = 0.05               # fraction of the servo limit
    # The localiser, whose pose the cart prediction starts from. These are the figures
    # amr_isaac publishes for the indoor beacons, and the same order as AMCL on this
    # map: a bias that lasts the run plus per-reading noise.
    localisation_xy_noise = 0.02            # m
    localisation_yaw_noise = 0.017          # rad
    localisation_xy_bias = 0.05             # m
    localisation_yaw_bias = 0.035           # rad

    sim: SimulationCfg = SimulationCfg(dt=1.0 / 200.0, render_interval=4)
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=2048, env_spacing=8.0,
                                                     replicate_physics=True)
    robot: ArticulationCfg = None    # built in __post_init__, needs the asset path

    # ---- actuators, from base.yaml / hitch.yaml -------------------------------------
    wheel_torque = 30.0             # Nm, ceiling for the per-episode limit below
    wheel_damping = 1000.0          # Nm s/rad, the driver's stiff velocity loop
    hitch_kp = 50.0
    hitch_kd = 2.0
    hitch_torque = 10.6             # Nm, XM540-W270 stall
    # The pitch servo's effort limit, which followed `hitch_torque` until 2026-10-23.
    # That made every hitch-sizing result a sweep of *both* joints: raising the yaw
    # limit raised the limit on the servo that holds the hook, and at 30 N.m that servo
    # presses the robot's tail down with about 559 N and moves load off the drive
    # wheels. None is "the same as the yaw limit", which is the old behaviour.
    pitch_torque: float | None = None
    hitch_passive_kd = 0.05         # bearing friction when the servo is off
    # Where the pitch servo holds the hook while towing. ``hook_up`` in
    # amr_docking/commands.py is 0.2 rad, but that is the angle that *catches* the bar:
    # holding it there once the bar is seated asks the servo to lift the front of the
    # cart off its wheels, which it cannot do, so it saturates for the whole run. Zero
    # is where the bar sits with the cart's wheels on the ground (the docking geometry
    # puts the hook tip at the bar height with the pitch level), so the servo carries
    # the bar and nothing else. See ``authority.py --pitch-targets 0.0 0.2``.
    pitch_target = 0.0

    # ---- randomisation --------------------------------------------------------------
    # The real robot has pulled 100 kg, so the sweep has to reach it. 22.6 kg is the
    # cart itself, so this is what goes on the deck.
    payload_mass_range = (0.0, 100.0)
    payload_x_range = (-0.25, 0.25)         # m in the cart frame
    payload_y_range = (-0.15, 0.15)
    payload_z_range = (0.30, 0.60)
    hitch_effort_range = (10.6, 10.6)       # Nm; widen to sweep the servo size
    # MDH300: 11 Nm rated, peak not published. An in-wheel BLDC usually peaks at two to
    # three times rated, and the real robot has pulled 100 kg, which 11 Nm does not
    # account for - so the range spans rated to a plausible peak rather than assuming
    # either. `authority.py --wheel-torques 40` measures the demand unclipped instead.
    wheel_effort_range = (11.0, 30.0)
    # Tyre against the floor. Sets how much traction is available, which is a different
    # question from how much is needed: on a smooth floor the motor runs out first, on a
    # slippery one the wheels slip first. Both carts' wheels get the same floor.
    floor_friction_range = (0.5, 1.0)
    # Rolling resistance as its coefficient c_rr, converted to a joint torque in
    # apply_cart. On Isaac Sim 5.x the joint friction figure is a resisting *effort*,
    # not a dimensionless coefficient, so the wheel load has to be supplied here rather
    # than left to the solver: the axle torque is c_rr * N_wheel * r_wheel. See the
    # comment in apply_cart for the measurement that settles which model is in force.
    cart_rolling_resistance_range = (0.01, 0.03)
    cart_axle_damping = 0.01                # Nm s/rad, bearing drag on top of that
    # Scrub friction at a caster swivel: the coefficient between tyre and floor, acting
    # at the contact patch radius and over the load that caster carries, so the swivel
    # torque is mu * N_caster * scrub_radius. This is the dominant unknown for the real
    # cart - it is what a caster resists being turned on the spot with, and the only
    # thing between the hitch servo and a cart it can swing freely. At 0.4-1.0 over a
    # 22.6-122.6 kg cart that is 0.27-3.6 N.m per caster, against the hitch's 10.6 N.m.
    # Four-caster cart only; the trailer resists with its rear axle.
    swivel_scrub_range = (0.4, 1.0)
    scrub_radius = 0.012                    # m, half the tyre contact patch
    beta_init_range = (-0.3, 0.3)           # rad

    # ---- goals ----------------------------------------------------------------------
    goal_distance_range = (1.0, 3.5)
    # Tolerances to match what the campaign actually achieves, not something tighter.
    # E5 stopped 254 +- 66 mm from the goal with the trailer cart and 183 +- 97 mm with
    # the four-caster one, so a 150 mm / 11 deg box asks the policy to beat the measured
    # baseline before it gets any credit - the reward then climbs while the success rate
    # stays near zero, which is what the first runs on the corrected model did.
    goal_position_tolerance = 0.25           # m
    goal_heading_tolerance = 0.35            # rad
    # The goal heading starts near the bearing to the goal (drive through it) and opens
    # out to anything over the first `curriculum_steps` control steps.
    curriculum_steps = 4_000_000
    heading_range_start = math.pi / 6
    jackknife = 1.45                         # rad, |beta| that ends the episode

    # ---- loading bay ----------------------------------------------------------------
    # The goal above is a pose on an empty floor, and that is the weakest part of the
    # experiment: nothing but the tolerance says the manoeuvre was precise, and a reader
    # is entitled to see "an actuator was switched off and the task got harder". A bay is
    # the job the application actually has - back the cart into a slot between two racks -
    # and there the *walls* set the precision, not a tolerance chosen by the author. Every
    # episode ends in a reverse, because a cart whose nose points out of the slot can only
    # have entered it backwards.
    #
    # Off by default. The free-space task keeps its recorded numbers and its checkpoints.
    bay = False
    # 1.00 m clear span against a 0.56 m deck: 220 mm a side. 0.80 m was tried first and
    # is too narrow to be the measurement it was meant to be --- with contact ending the
    # episode, 120 mm a side is inside the spread of the manoeuvre itself (the clearance
    # of the reaches that did get in had an s.d. of 144 mm), so every condition fails
    # almost always and the comparison says nothing. Widened so that the walls are a
    # constraint the policy can respect rather than one it cannot.
    bay_width = 1.00                 # m, clear span between the side walls; deck is 0.56
    bay_depth = 1.6                  # m, mouth to the inner face of the back wall
    bay_wall_thickness = 0.1
    bay_wall_height = 0.5
    # Where the episode starts. The distance is from the mouth to the *robot*, and the
    # cart is a lever behind it, so the cart starts 0.5-2.5 m outside.
    start_distance_range = (1.5, 3.5)
    # Curriculum: the first episodes start square in front of the mouth, which is a
    # straight reverse, and the approach opens out over `curriculum_steps`. Without it
    # every episode needs the reverse from the first step and the policy has nothing to
    # climb - the same reason the free-space goal heading opens out.
    start_spread_start = 0.25        # rad
    start_spread_end = 1.2           # rad
    # Its own schedule length, and not `curriculum_steps`, for a reason worth stating.
    # `common_step_counter` advances once per *control step of the whole batch*, so a
    # 3000-iteration run at 32 steps per iteration advances it 96,000 times in total --
    # not 196 million, which is the number of environment steps. The free-space task's
    # `curriculum_steps = 4_000_000` therefore never completes: it reaches 2.4% of its
    # schedule, which is why that task's goal-heading window opens only to +-34 deg and
    # why the paper reports that as a limitation rather than as a design. The bay
    # approach must not inherit the same mistake, so its schedule is set in the units
    # the counter actually uses and finishes at the half-way point of training.
    bay_curriculum_steps = 48_000
    # Wall contact is read off the geometry rather than from a contact sensor, for the
    # reason corridor_clearance.py recomputes the ROS campaign's clearance from the
    # trajectory (trap 25): the walls are static and their pose is exact, so the deck
    # corners give the gap directly and a threshold on it is reproducible. The ROS
    # measurement needed a 0.5 s persistence rule because its source was a noisy beacon;
    # this one has no noise to reject.
    bay_contact_clearance = 0.01     # m
    # Touching a wall ends the episode, and is a failure.
    #
    # The first version of this task measured the clearance and did nothing with it:
    # `min_clearance` reached neither the reward nor the termination, so the success
    # criterion was still the pose tolerance and the walls set nothing. A policy is not
    # wrong to exploit that, and it did --- of 693 reached poses at least 255 (36.8%)
    # got there with the deck in contact with a wall, mean clearance -11 mm. The task
    # was described as one where "the walls set the precision, not a tolerance chosen by
    # the author" and that description was false of the implementation.
    #
    # So contact terminates, which makes the reported metric and the trained objective
    # the same thing, and a shaping term gives a gradient before contact rather than a
    # cliff at it.
    bay_wall_margin = 0.06           # m, where the shaping penalty starts

    # ---- reward weights -------------------------------------------------------------
    w_progress = 8.0
    w_near = 2.0
    w_heading = 2.0
    w_success = 60.0
    w_beta = -4.0
    w_effort = -0.05
    w_action_rate = -0.05
    w_jackknife = -15.0
    # Contact is as terminal as a jackknife, and weighted the same, because both end the
    # episode having damaged something the application cares about.
    w_wall = -8.0                    # shaping, per second, at the wall
    w_contact = -15.0                # impulse on the step contact ends the episode
    beta_soft_limit = 1.2

    # ---- evaluation protocol ---------------------------------------------------------
    # Both of these are off during training and exist so that an evaluation can be the
    # measurement it is reported as.
    #
    # `curriculum` pins both curricula to a point on their schedule instead of reading
    # `common_step_counter`. The counter is zero in a freshly constructed environment and
    # advances once per control step of the batch, so an evaluation reads the curriculum
    # at the *start* of its schedule however long the policy trained: a bay evaluation of
    # a policy that finished on +-1.2 rad of approach was sampling +-0.26 rad, the
    # easiest twentieth of its own training distribution. 1.0 evaluates on the
    # distribution training ended with.
    curriculum: float | None = None
    # `scenario_seed` makes an episode a lookup rather than a draw. Two conditions
    # otherwise never see the same episodes - terminations differ, so resets fall on
    # different steps and the stream diverges from the first one - and a paired
    # comparison needs them to. Environment i's k-th episode is scenario
    # k * num_envs + i, and each scenario's numbers come from a generator seeded by its
    # own index, so neither the order resets fall in nor how many reset together can
    # change what a scenario is.
    scenario_seed: int | None = None

    def __post_init__(self):
        self.sim.render_interval = self.decimation
        if not self.active_hitch:
            self.action_space = 2
        # Odometry (2, or 3 with the unmeasurable lateral velocity) + hitch angle, rate
        # and torque (3) + the bar angle if it is being given away (1) + goal (4), then
        # the commanded hitch angle if there is one, then the previous action.
        self.obs_per_frame = ((9 if self.observable_only else 11)
                              + self.action_space + (1 if self.active_hitch else 0))
        self.observation_space = self.history * self.obs_per_frame
        # True cart pose error (4), cart velocity (3), the randomised parameters (9),
        # and this episode's sensor calibration (7) when there is one to report. With the
        # noise model off those seven are constant, so they are left out - which also
        # keeps a noiseless run loadable by a checkpoint trained before they existed.
        self.privileged = 16 + (7 if self.sensor_noise else 0)
        self.state_space = self.observation_space + self.privileged


# Columns of the per-scenario random numbers used when `scenario_seed` is set. The
# quantities take *fixed* columns rather than consecutive ones, because the branches in
# the reset differ between conditions - a locked hitch has no initial angle to sample -
# and a running counter would then hand the same scenario a different payload under one
# condition than under another, which is the whole thing the lookup is for.
SCENARIO_SLOTS = 32


def _deck_corners(pos, yaw):
    """The four deck corners in the bay frame, as (n, 4) tensors of x and y."""
    cs, sn = torch.cos(yaw), torch.sin(yaw)
    xs, ys = [], []
    # Cycle order, not the product order: `_bay_gap` forms the deck edges by pairing
    # each corner with the next one, so corner i and corner i+1 must be adjacent.
    hl, hw = asset.DECK_HALF_L, asset.DECK_HALF_W
    for ex, ey in ((hl, hw), (hl, -hw), (-hl, -hw), (-hl, hw)):
        xs.append(pos[:, 0] + ex * cs - ey * sn)
        ys.append(pos[:, 1] + ex * sn + ey * cs)
    return torch.stack(xs, dim=1), torch.stack(ys, dim=1)


def _bay_gap(px, py, cfg):
    """Smallest distance between the deck rectangle and any of the three wall solids.

    Both are convex polygons, and the minimum distance between two disjoint convex
    polygons is attained at a vertex of at least one of them. So the gap is the smaller
    of two families: each deck corner against each wall box, and each wall corner
    against the deck.

    The second family is not an embellishment. An earlier version took the first alone,
    on the argument that "the nearest point of a convex polygon to a plane is a vertex".
    The walls are boxes, not planes, and a box has corners: a deck *edge* can lie across
    the front corner of a rack with all four deck corners well outside the wall. Over
    4 million sampled deck poses restricted to ones the colliders allow, the corner-only
    form overstated the gap in 42.7% of poses and reported at least 10 mm of clearance
    while the deck was in fact touching a wall in 0.84% of them --- 2.33% of the poses
    within 200 mm of a wall, where it matters --- with a median reported clearance of
    79 mm. That is a missed contact, and contact both terminates the episode and feeds
    the shaping term.

    Overlap is detected separately, by the separating-axis test, and reported as minus
    the penetration depth. Neither vertex family catches it on its own: two rectangles
    can cross in a plus shape with no vertex of either inside the other, and over 4
    million sampled poses 11.8% are of that kind. They are all deep overlaps that the
    colliders do not permit, and reaching one continuously from outside passes through
    tangency, where a vertex of one polygon lies on the other and the vertex families
    read exactly zero, so the 10 mm contact test fires first either way. The test is
    here so that the function is correct without that argument having to hold.

    `px`, `py` are (n, 4) and in cycle order, so corner i and corner i+1 are adjacent.
    The result is (n,).
    """
    t, w, d = cfg.bay_wall_thickness, cfg.bay_width, cfg.bay_depth
    # (centre_x, centre_y, half_x, half_y) for the left wall, the right wall, the back.
    boxes = ((-0.5 * d, 0.5 * (w + t), 0.5 * d, 0.5 * t),
             (-0.5 * d, -0.5 * (w + t), 0.5 * d, 0.5 * t),
             (-d - 0.5 * t, 0.0, 0.5 * t, 0.5 * (w + 2 * t)))
    best = None
    for cx, cy, hx, hy in boxes:
        dx = (px - cx).abs() - hx
        dy = (py - cy).abs() - hy
        outside = torch.sqrt(torch.clamp(dx, min=0.0) ** 2
                             + torch.clamp(dy, min=0.0) ** 2)
        inside = torch.clamp(torch.maximum(dx, dy), max=0.0)
        sdf = (outside + inside).min(dim=1).values
        best = sdf if best is None else torch.minimum(best, sdf)

    # The other family: every wall corner against the deck rectangle.
    corners = torch.tensor(
        [[cx + sx * hx, cy + sy * hy]
         for cx, cy, hx, hy in boxes for sx in (-1.0, 1.0) for sy in (-1.0, 1.0)],
        device=px.device, dtype=px.dtype)                      # (12, 2)
    ax, ay = px.unsqueeze(2), py.unsqueeze(2)                  # (n, 4, 1)
    ex = px.roll(-1, dims=1).unsqueeze(2) - ax                 # edge vectors
    ey = py.roll(-1, dims=1).unsqueeze(2) - ay
    qx, qy = corners[:, 0], corners[:, 1]                      # (12,)
    rx, ry = qx - ax, qy - ay                                  # (n, 4, 12)
    span = torch.clamp(ex * ex + ey * ey, min=1e-12)
    u = torch.clamp((rx * ex + ry * ey) / span, 0.0, 1.0)
    dist = torch.sqrt(torch.clamp((rx - u * ex) ** 2 + (ry - u * ey) ** 2, min=0.0))
    cross = ex * ry - ey * rx                                  # side of each edge
    within = (cross >= 0).all(dim=1) | (cross <= 0).all(dim=1)  # (n, 12)
    signed = torch.where(within, -dist.min(dim=1).values, dist.min(dim=1).values)
    gap = torch.minimum(best, signed.min(dim=1).values)

    # Separating-axis test. Four axes suffice for two rectangles: the box's two, and
    # the deck's two edge normals (opposite edges share an axis). On every axis the
    # overlap depth is min(max) - max(min); the smallest positive one over the axes is
    # the penetration depth, and any non-positive one means the two are apart.
    ex2 = px.roll(-1, dims=1) - px
    ey2 = py.roll(-1, dims=1) - py
    span = torch.sqrt(torch.clamp(ex2 * ex2 + ey2 * ey2, min=1e-12))
    axes = [(torch.ones_like(px[:, :1]), torch.zeros_like(px[:, :1])),
            (torch.zeros_like(px[:, :1]), torch.ones_like(px[:, :1])),
            (-ey2[:, :1] / span[:, :1], ex2[:, :1] / span[:, :1]),
            (-ey2[:, 1:2] / span[:, 1:2], ex2[:, 1:2] / span[:, 1:2])]
    for cx, cy, hx, hy in boxes:
        corner_x = px.new_tensor([cx - hx, cx + hx, cx + hx, cx - hx])
        corner_y = px.new_tensor([cy - hy, cy - hy, cy + hy, cy + hy])
        depth = None
        for ax, ay in axes:
            a = (px * ax + py * ay)                        # deck on this axis
            b = corner_x * ax + corner_y * ay              # box on this axis
            d = torch.minimum(a.max(dim=1).values, b.max(dim=1).values) \
                - torch.maximum(a.min(dim=1).values, b.min(dim=1).values)
            depth = d if depth is None else torch.minimum(depth, d)
        gap = torch.where(depth > 0.0, torch.minimum(gap, -depth), gap)
    return gap


def make_robot_cfg(cfg: TowingEnvCfg, usd_path: str) -> ArticulationCfg:
    """Articulation config for the built asset: velocity wheels, a position hitch."""
    # Locked and active use the same servo gains; what differs is whether the policy
    # moves the target. Passive turns the servo off and leaves the bearing drag.
    driven = cfg.active_hitch or cfg.lock_hitch or cfg.rigid_lock
    kw = dict(joint_names_expr=['hitch_yaw_joint'],
              stiffness=cfg.hitch_kp if driven else 0.0,
              damping=cfg.hitch_kd if driven else cfg.hitch_passive_kd,
              effort_limit=cfg.hitch_torque)
    hitch_yaw = (DCMotorCfg(saturation_effort=cfg.hitch_torque,
                            velocity_limit=cfg.hitch_noload_speed, **kw)
                 if cfg.hitch_dc else ImplicitActuatorCfg(**kw))
    return ArticulationCfg(
        prim_path='/World/envs/env_.*/Robot',
        spawn=sim_utils.UsdFileCfg(
            usd_path=usd_path,
            activate_contact_sensors=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False, max_depenetration_velocity=1.0,
                linear_damping=0.0, angular_damping=0.0),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=cfg.solver_position_iterations,
                solver_velocity_iteration_count=cfg.solver_velocity_iterations)),
        # The articulation root is base_link, so this is base_footprint on the floor.
        # All joints default to zero, which is the pose the asset is authored in; the
        # towing pose (hook raised, bar turned back to match) is set per episode in
        # _reset_idx, where it can be written together with the hitch angle.
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, asset.BASE_Z)),
        actuators={
            'wheels': ImplicitActuatorCfg(
                joint_names_expr=['.*_wheel_joint'], stiffness=0.0,
                damping=cfg.wheel_damping, effort_limit=cfg.wheel_torque),
            'hitch_yaw': hitch_yaw,
            'hitch_pitch': ImplicitActuatorCfg(
                joint_names_expr=['hitch_pitch_joint'], stiffness=cfg.hitch_kp,
                damping=cfg.hitch_kd,
                effort_limit=(cfg.hitch_torque if cfg.pitch_torque is None
                              else cfg.pitch_torque)),
            # Free joints: no stiffness, only the viscous term that stands for bearing
            # friction and rolling resistance. The four-caster cart has swivels as well;
            # the trailer has none, and an expression that matches nothing is an error.
            'passive': ImplicitActuatorCfg(
                joint_names_expr=(['tow_bar_joint', 'cart_axle_.*']
                                  + (['cart_swivel_.*'] if cfg.cart_wheels == 'caster4'
                                     else [])),
                stiffness=0.0, damping=cfg.cart_axle_damping,
                effort_limit=50.0),
        })


class TowingEnv(DirectRLEnv):
    cfg: TowingEnvCfg

    def __init__(self, cfg: TowingEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        n, dev = self.num_envs, self.device

        self._wheels = self.robot.find_joints(['left_wheel_joint', 'right_wheel_joint'])[0]
        self._yaw = self.robot.find_joints('hitch_yaw_joint')[0][0]
        self._pitch = self.robot.find_joints('hitch_pitch_joint')[0][0]
        self._bar = self.robot.find_joints('tow_bar_joint')[0][0]
        self._axles = self.robot.find_joints(['cart_axle_.*'])[0]
        self._swivels = (self.robot.find_joints(['cart_swivel_.*'])[0]
                         if cfg.cart_wheels == 'caster4' else [])
        self._base = self.robot.find_bodies('base_link')[0][0]
        self._cart = self.robot.find_bodies('cart_body')[0][0]

        if cfg.rigid_lock:
            # Pinned position limits make the yaw joint a constraint the solver enforces,
            # so no load can drag it off zero. Writing them once is enough: limits are
            # not touched by a reset.
            #
            # The call writes the whole limit buffer, so PhysX logs
            # "setLimitParams() only supports limit angles in range [-2Pi, 2Pi]" for the
            # eleven continuous joints, whose limits are FLT_MAX. Those writes are
            # rejected and the joints stay free, which is what is wanted: checked by
            # reading the limits back (only hitch_yaw_joint is at +-0.002) and by
            # writing 2.0 rad to the caster swivels, which holds.
            play = cfg.rigid_lock_play
            self.robot.write_joint_position_limit_to_sim(
                torch.tensor([[[-play, play]]], device=dev).repeat(n, 1, 1),
                joint_ids=[self._yaw], warn_limit_violation=False)

        self.actions = torch.zeros(n, self.cfg.action_space, device=dev)
        self.prev_actions = torch.zeros_like(self.actions)
        self.obs_history = torch.zeros(n, self.cfg.history, self.cfg.obs_per_frame,
                                       device=dev)
        self.goal = torch.zeros(n, 3, device=dev)       # x, y, heading in world
        self.prev_distance = torch.zeros(n, device=dev)
        self.hitch_torque = torch.zeros(n, device=dev)
        # The torque ceiling in force this step: the flat limit, or the
        # torque-speed bound when the DC curve is used.
        self.hitch_ceiling = torch.full((n,), self.cfg.hitch_torque, device=dev)
        self.pitch_torque = torch.zeros(n, device=dev)
        self.beta_target = torch.zeros(n, device=dev)
        # The pose buffers still hold the finished episode after a reset, but `goal` does
        # not - it has already been resampled. So the terminal error is recorded while
        # the two still agree, in _get_dones, and read from here afterwards.
        self.last_distance = torch.zeros(n, device=dev)
        self.last_yaw_error = torch.zeros(n, device=dev)
        self.last_reached = torch.zeros(n, dtype=torch.bool, device=dev)
        self.last_jackknife = torch.zeros(n, dtype=torch.bool, device=dev)
        self.last_beta = torch.zeros(n, device=dev)
        # Smallest gap from a deck corner to a bay wall, this episode and the one that
        # just ended. Far away when there is no bay, so a reader of the table can tell
        # the free-space task from the bay one by the column being empty.
        self.min_clearance = torch.full((n,), 10.0, device=dev)
        self.last_clearance = torch.full((n,), 10.0, device=dev)
        # This step's gap, which the reward reads, as opposed to the episode's smallest,
        # which the report reads. 10 m is "not in the slot", so no penalty applies there.
        self.wall_gap = torch.full((n,), 10.0, device=dev)
        self.entered = torch.zeros(n, dtype=torch.bool, device=dev)
        self.last_entered = torch.zeros(n, dtype=torch.bool, device=dev)
        self.last_contact = torch.zeros(n, dtype=torch.bool, device=dev)
        # The other two ways an episode can end, kept so that a failure can be given a
        # cause. Without them a report can say an episode did not reach the goal but not
        # whether it ran out of time, tipped the cart or was ended by a wall.
        self.last_tipped = torch.zeros(n, dtype=torch.bool, device=dev)
        self.last_timeout = torch.zeros(n, dtype=torch.bool, device=dev)
        # Which episode each environment is on, and the scenario the current one came
        # from, for `scenario_seed`. The id is -1 while the draw is random.
        self.episode_index = torch.zeros(n, dtype=torch.long, device=dev)
        self.scenario_id = torch.full((n,), -1, dtype=torch.long, device=dev)
        self.success_rate = 0.0
        self.jackknife_rate = 0.0
        self.contact_rate = 0.0
        self.hitch_limit = torch.full((n,), self.cfg.hitch_torque, device=dev)
        # payload mass, x, y, z; hitch limit; rolling resistance; swivel scrub;
        # wheel limit; floor friction
        self.params = torch.zeros(n, 9, device=dev)
        self.scale = torch.tensor(self.cfg.action_scale[:self.cfg.action_space], device=dev)
        # This episode's sensor calibration, which the critic sees and the actor does
        # not: odometry scale on v and omega, the hitch zero offset, the localiser's bias
        # in x, y and yaw, and the actuation delay.
        self.bias = torch.zeros(n, 7, device=dev)
        self.bias[:, :2] = 1.0
        self.latency = torch.zeros(n, dtype=torch.long, device=dev)
        # Commands wait their delay out here; index 0 is the newest.
        self._delay_max = (int(self.cfg.latency_steps_range[1])
                           if self.cfg.sensor_noise else 0)
        self._queue = torch.zeros(n, self._delay_max + 1, self.cfg.action_space,
                                  device=dev)
        self.applied = torch.zeros_like(self.actions)

        # Body masses, centres of mass and inertias are randomised per reset, so keep the
        # authored values to build each episode's from.
        view = self.robot.root_physx_view
        self._default_masses = view.get_masses().clone()
        self._default_coms = view.get_coms().clone()
        self._default_inertias = view.get_inertias().clone()
        # PhysX takes the whole buffer and a row index, not just the rows that changed,
        # so these are kept and written back each reset rather than rebuilt.
        self._masses = self._default_masses.clone()
        self._coms = self._default_coms.clone()
        self._inertias = self._default_inertias.clone()
        self._materials = view.get_material_properties().clone()
        # Which collision shapes are tyres. The asset deliberately gives different
        # materials to different shapes - the support casters are frictionless, which is
        # what makes the trailer a trailer - so the floor friction must be written to the
        # tyres alone. PhysX indexes materials by shape, not by body, and neither the
        # articulation nor its view reports shapes per body, so count them per link.
        shapes, self._tyre_shapes = [], []
        for path in view.link_paths[0]:
            shapes.append(self.robot._physics_sim_view
                          .create_rigid_body_view(path).max_shapes)
        tyres = [i for i, name in enumerate(self.robot.body_names)
                 if name.endswith('_wheel_link') or name.startswith('cart_wheel_')]
        for body in tyres:
            start = sum(shapes[:body])
            self._tyre_shapes.extend(range(start, start + shapes[body]))
        if sum(shapes) != view.max_shapes or not self._tyre_shapes:
            raise RuntimeError(f'could not locate the tyre collision shapes: '
                               f'{sum(shapes)} counted vs {view.max_shapes} reported')

    # ---- scene -----------------------------------------------------------------------
    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot)
        # A static box rather than Isaac's GroundPlaneCfg, which spawns a USD fetched
        # from S3: a run should not need the network, and one sweep did die on
        # "Unable to open the usd file at path: https://omniverse-content-production...".
        # The floor is left at a high friction and the tyres carry the randomised
        # coefficient, since both materials combine with 'min'.
        ground = sim_utils.CuboidCfg(
            size=(400.0, 400.0, 0.4),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0, dynamic_friction=1.0, restitution=0.0,
                friction_combine_mode='min', restitution_combine_mode='min'),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.3, 0.3, 0.32)))
        ground.func('/World/ground', ground, translation=(0.0, 0.0, -0.2))
        if self.cfg.backdrop:
            self._add_backdrop()
        if self.cfg.bay:
            self._add_bay()
        self.scene.clone_environments(copy_from_source=False)
        if self.cfg.bay:
            # The walls are cloned into every environment, so without this a robot that
            # wanders far enough sideways collides with its neighbour's bay instead of
            # its own. Only for the bay task: the free-space floor has nothing on it to
            # collide with, and its recorded results were produced without this call.
            self.scene.filter_collisions(global_prim_paths=['/World/ground'])
        self.scene.articulations['robot'] = self.robot
        light = sim_utils.DomeLightCfg(intensity=2000.0)
        light.func('/World/light', light)

    @staticmethod
    def _add_backdrop(path='/World/env'):
        """Reference the warehouse in as scenery, two centimetres below the floor.

        The building has colliders of its own, and its floor is at z = 0 like ours. Two
        coplanar floors would both make contacts, so a tyre would sometimes land on the
        warehouse material instead of this task's randomised one and the episode would
        not be the one the figure was drawn from. Dropping the whole reference 20 mm puts
        its floor out of reach while leaving the walls and racking where they look right;
        they are tens of metres away, well outside a 20 s episode.
        """
        from isaacsim.core.utils.nucleus import get_assets_root_path
        from isaacsim.core.utils.stage import add_reference_to_stage, get_current_stage
        try:
            root = get_assets_root_path()
        except RuntimeError as e:        # the asset server is not reachable
            print(f'[towing_env] no backdrop, asset server: {e}', flush=True)
            return
        if not root:
            print('[towing_env] no backdrop, asset server unreachable', flush=True)
            return
        add_reference_to_stage(
            root + '/Isaac/Environments/Simple_Warehouse/warehouse.usd', path)
        prim = get_current_stage().GetPrimAtPath(path)
        xf = UsdGeom.Xformable(prim)
        xf.ClearXformOpOrder()
        xf.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, -0.02))

    def _add_bay(self, path='/World/envs/env_0/Bay'):
        """Spawn two side walls and a back wall as static boxes in the environment template.

        The slot runs from the mouth at x = 0 back to x = -bay_depth, centred on y = 0,
        so the goal pose below is the slot's own centre and the approach is from +x.
        Spawned before `clone_environments`, which replicates them into every
        environment; static colliders like the floor, not rigid bodies, because nothing
        in the task should be able to push a rack out of the way.
        """
        c = self.cfg
        t, h = c.bay_wall_thickness, c.bay_wall_height
        material = sim_utils.RigidBodyMaterialCfg(
            static_friction=0.8, dynamic_friction=0.8, restitution=0.0,
            friction_combine_mode='min', restitution_combine_mode='min')

        def wall(name, size, pos):
            box = sim_utils.CuboidCfg(
                size=size,
                collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
                physics_material=material,
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.42, 0.40, 0.37)))
            box.func(f'{path}/{name}', box, translation=pos)

        # Inner faces at +-bay_width/2 and at -bay_depth, so the clear span is the
        # figure in the cfg and the clearance below is measured against it.
        for name, sign in (('left', 1.0), ('right', -1.0)):
            wall(name, (c.bay_depth, t, h),
                 (-0.5 * c.bay_depth, sign * 0.5 * (c.bay_width + t), 0.5 * h))
        wall('back', (t, c.bay_width + 2 * t, h),
             (-c.bay_depth - 0.5 * t, 0.0, 0.5 * h))

    def show_goal(self, env_index=0):
        """Draw the commanded cart pose on the floor, for a recording.

        A clip of the robot does not show where the cart was asked to end up, so a
        viewer cannot tell why one run is a success and another a timeout. This draws
        the deck footprint at the goal with an arrow for its heading.

        Visual only, and built the first time it is called, outside the environment
        template so `clone_environments` never sees it: no training or evaluation run
        spawns it, and nothing in the task reads it.
        """
        stage = self.sim.stage
        root = '/World/GoalMarker'
        if not stage.GetPrimAtPath(root).IsValid():
            UsdGeom.Xform.Define(stage, root)
            amber = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.95, 0.42, 0.16),
                                                emissive_color=(0.55, 0.18, 0.03),
                                                roughness=0.6)
            bar, h, z = 0.045, 0.012, 0.006
            lx, ly = 0.84, 0.56          # the cart deck, as docking_targets.yaml has it

            def box(name, size, pos, quat=None):
                cfg = sim_utils.CuboidCfg(size=size, visual_material=amber)
                cfg.func(f'{root}/{name}', cfg, translation=pos, orientation=quat)

            # The deck footprint as four bars rather than a filled slab: a solid patch
            # on the floor reads as an obstacle, an outline reads as a target.
            box('front', (bar, ly, h), (0.5 * lx - 0.5 * bar, 0.0, z))
            box('back', (bar, ly, h), (-0.5 * lx + 0.5 * bar, 0.0, z))
            for name, sign in (('left', 1.0), ('right', -1.0)):
                box(name, (lx - 2 * bar, bar, h),
                    (0.0, sign * (0.5 * ly - 0.5 * bar), z))
            # Heading arrow, ahead of the footprint. The cone's axis is +z, so it is
            # laid along +x by a quarter turn about y.
            box('shaft', (0.30, bar, h), (0.5 * lx + 0.15, 0.0, z))
            head = sim_utils.ConeCfg(radius=0.075, height=0.18, visual_material=amber)
            head.func(f'{root}/head', head,
                      translation=(0.5 * lx + 0.39, 0.0, z),
                      orientation=(0.70710678, 0.0, 0.70710678, 0.0))
        prim = stage.GetPrimAtPath(root)
        gx, gy, gyaw = (float(v) for v in self.goal[env_index, :3])
        origin = self.scene.env_origins[env_index]
        xf = UsdGeom.Xformable(prim)
        xf.ClearXformOpOrder()
        xf.AddTranslateOp().Set(Gf.Vec3d(gx + float(origin[0]),
                                         gy + float(origin[1]),
                                         float(origin[2])))
        xf.AddRotateZOp().Set(math.degrees(gyaw))

    def _track_clearance(self, pos, yaw):
        """Keep this episode's smallest deck-corner-to-wall gap, and this step's.

        The gap is the true distance from the deck to the wall solids, not the lateral
        offset from the slot's centreline. The first version of this took the second,
        counting any corner with px < 0 as held by the side walls and reading its gap as
        bay_width/2 - |py|. For a corner between the walls those agree, but for a corner
        that crosses the mouth plane *beside* the bay --- free floor, outside the walls
        altogether --- the lateral form reports deep penetration: a corner at y = 0.9 m
        with the slot half-span at 0.5 m came out as -400 mm. The gap then jumped from
        the +10 m sentinel to -0.4 m in a single step.

        While the figure was only reported that was a wrong number. Once contact
        terminated the episode and a shaping term read the same figure, it became a
        -38 reward spike for driving *past* the bay, which is the manoeuvre the active
        policy uses most --- it swings the cart wide before reversing. The active
        condition learned to stay away from the bay entirely (29 of 1024 episodes got
        the deck into the slot, 0 reached the goal) while the passive one, which keeps
        the cart closer to the robot's own path, was punished far less often and learned
        the task. The comparison inverted, for a reason that was in the metric.
        """
        self._corner_px, self._corner_py = _deck_corners(pos, yaw)
        gap = _bay_gap(self._corner_px, self._corner_py, self.cfg)
        self.wall_gap = gap
        self.min_clearance = torch.minimum(self.min_clearance, gap)
        self.last_clearance = self.min_clearance.clone()
        # Whether the deck has ever been between the walls, which the report needs and
        # the gap no longer carries: the gap is finite and positive out on the floor, so
        # there is no sentinel value left to stand for "never entered the slot".
        inside = (self._corner_px < 0.0) & \
            (self._corner_py.abs() < 0.5 * self.cfg.bay_width)
        self.entered = self.entered | inside.any(dim=1)
        # Cloned for the same reason last_clearance is: the reset that follows a done in
        # the same step clears `entered` before the report can read it.
        self.last_entered = self.entered.clone()

    # ---- stepping --------------------------------------------------------------------
    def _pre_physics_step(self, actions: torch.Tensor):
        self.prev_actions = self.actions.clone()
        self.actions = torch.clamp(actions, -1.0, 1.0)
        # What the robot does is what reached it, which is a few tens of milliseconds
        # old; what the policy observes itself doing is what it commanded. Keeping the
        # two apart is the whole point: a policy handed the delayed value back has no
        # delay left to learn to handle.
        if self._delay_max:
            self._queue = torch.roll(self._queue, shifts=1, dims=1)
            self._queue[:, 0] = self.actions
            index = self.latency.view(-1, 1, 1).expand(-1, 1, self.cfg.action_space)
            self.applied = self._queue.gather(1, index).squeeze(1)
        else:
            self.applied = self.actions

    def _apply_action(self):
        a = self.applied * self.scale
        v, omega = a[:, 0], a[:, 1]
        left = (v - omega * WHEEL_SEPARATION / 2) / WHEEL_RADIUS
        right = (v + omega * WHEEL_SEPARATION / 2) / WHEEL_RADIUS
        self.robot.set_joint_velocity_target(torch.stack((left, right), dim=1),
                                             joint_ids=self._wheels)
        pitch = torch.full((self.num_envs, 1), self.cfg.pitch_target, device=self.device)
        if self.cfg.lock_hitch:
            self.robot.set_joint_position_target(
                torch.cat((self.beta_target.unsqueeze(1), pitch), dim=1),
                joint_ids=[self._yaw, self._pitch])
        elif self.cfg.active_hitch:
            limit = self.cfg.beta_command_limit
            self.beta_target = torch.clamp(self.beta_target + a[:, 2] * self.physics_dt,
                                           -limit, limit)
            self.robot.set_joint_position_target(
                torch.cat((self.beta_target.unsqueeze(1), pitch), dim=1),
                joint_ids=[self._yaw, self._pitch])
        else:
            self.robot.set_joint_position_target(pitch, joint_ids=[self._pitch])
        # The servo torque the Dynamixel current reflects: Kp * error - Kd * rate, within
        # the servo's limit. Computed the same way as in amr_isaac/amr_sim.py so the
        # observation means the same thing on the real robot. With the servo off, only
        # the bearing drag is left.
        q = self.robot.data.joint_pos[:, self._yaw]
        qd = self.robot.data.joint_vel[:, self._yaw]
        if self.cfg.active_hitch or self.cfg.lock_hitch:
            tau = self.cfg.hitch_kp * (self.beta_target - q) - self.cfg.hitch_kd * qd
        else:
            tau = -self.cfg.hitch_passive_kd * qd
        if self.cfg.hitch_dc:
            # The four-quadrant bound DCMotorCfg actually applies, so the reported
            # torque is the torque the joint receives. A symmetric ceiling on |qd| was
            # used here before and is wrong: it attenuates the braking side, which is
            # the side this joint mostly works on, so it under-reports the applied
            # torque at speed and makes large reported torques look as though they
            # occur only near standstill.
            slip = qd / self.cfg.hitch_noload_speed
            hi = torch.minimum(self.hitch_limit * (1.0 - slip), self.hitch_limit)
            lo = torch.maximum(self.hitch_limit * (-1.0 - slip), -self.hitch_limit)
            self.hitch_torque = torch.clamp(tau, lo, hi)
            # What counts as "near the limit" is the ceiling in force at this speed,
            # not the flat one; comparing an attenuated signal against the flat limit
            # measures the reporting rather than the joint.
            self.hitch_ceiling = torch.maximum(hi, -lo)
        else:
            self.hitch_torque = torch.clamp(tau, -self.hitch_limit, self.hitch_limit)
            self.hitch_ceiling = self.hitch_limit
        # The pitch servo the same way. It matters more than it looks: the hook is held
        # at an angle, and whatever torque that takes is reacted into the robot's tail
        # through a 54 mm lever, 269 mm behind the wheel axis. A saturated pitch servo
        # therefore presses the tail down with ~200 N, which transfers weight off the
        # drive wheels and onto the (frictionless) support caster. That is how a towing
        # robot in simulation ends up unable to move a cart the real one pulls.
        qp = self.robot.data.joint_pos[:, self._pitch]
        qpd = self.robot.data.joint_vel[:, self._pitch]
        self.pitch_torque = torch.clamp(
            self.cfg.hitch_kp * (self.cfg.pitch_target - qp) - self.cfg.hitch_kd * qpd,
            -self.cfg.hitch_torque, self.cfg.hitch_torque)

    # ---- geometry --------------------------------------------------------------------
    def _base_pose(self):
        """base_footprint position and yaw in the environment's own frame."""
        pos = self.robot.data.body_link_pos_w[:, self._base] - self.scene.env_origins
        yaw = euler_xyz_from_quat(self.robot.data.body_link_quat_w[:, self._base])[2]
        return pos[:, :2], wrap_to_pi(yaw)

    def _cart_pose_true(self):
        pos = self.robot.data.body_link_pos_w[:, self._cart] - self.scene.env_origins
        yaw = euler_xyz_from_quat(self.robot.data.body_link_quat_w[:, self._cart])[2]
        return pos[:, :2], wrap_to_pi(yaw)

    def _cart_pose_predicted(self):
        """Predict the cart pose from the true robot pose and the true hitch angle."""
        base, yaw = self._base_pose()
        return self._predict_cart(base, yaw, self.robot.data.joint_pos[:, self._yaw])

    def _predict_cart(self, base, yaw, beta):
        """Predict the cart pose from a robot pose and a hitch angle alone.

        Exact for a single-axle trailer, because then the hitch angle *is* the cart's
        heading relative to the robot. Wrong for the four-caster cart, on purpose. The
        actor passes its *measured* pose and angle, so the localiser's error and the
        servo's zero offset land in the cart pose exactly as they will on the robot.
        """
        cart_yaw = wrap_to_pi(yaw + beta)
        # yaw axis on the robot, then the hook and the cart centre along the cart's axis
        ax = torch.stack((torch.cos(yaw), torch.sin(yaw)), dim=1)
        cx = torch.stack((torch.cos(cart_yaw), torch.sin(cart_yaw)), dim=1)
        pos = base + ax * YAW_AXIS_X + cx * CART_LEVER
        return pos, cart_yaw

    def _goal_error(self, pos, yaw):
        """Goal in the cart's own frame: dx, dy, cos/sin of the heading error."""
        delta = self.goal[:, :2] - pos
        c, s = torch.cos(-yaw), torch.sin(-yaw)
        dx = c * delta[:, 0] - s * delta[:, 1]
        dy = s * delta[:, 0] + c * delta[:, 1]
        dtheta = wrap_to_pi(self.goal[:, 2] - yaw)
        return dx, dy, dtheta

    # ---- observations ----------------------------------------------------------------
    def _read(self, value, noise, quantum):
        """Return the true value with the device's noise, in the device's steps."""
        value = value + torch.randn_like(value) * noise
        return torch.round(value / quantum) * quantum if quantum else value

    def _frame(self):
        d = self.robot.data
        c = self.cfg
        # Measured the way the robot measures: wheel odometry reports forward speed and
        # yaw rate and nothing else, the Dynamixel reports its own position, velocity and
        # current, and the localiser reports the robot's pose - from which the cart's is
        # predicted through the hitch angle.
        v = d.root_lin_vel_b[:, 0:1]
        w = d.root_ang_vel_b[:, 2:3]
        beta = d.joint_pos[:, self._yaw:self._yaw + 1]
        beta_dot = d.joint_vel[:, self._yaw:self._yaw + 1]
        tau = (self.hitch_torque / c.hitch_torque).unsqueeze(1)
        base, yaw = self._base_pose()
        if c.sensor_noise:
            v = self._read(v * self.bias[:, 0:1], c.odom_v_noise, c.odom_v_quantum)
            w = self._read(w * self.bias[:, 1:2], c.odom_w_noise, c.odom_w_quantum)
            beta = self._read(beta + self.bias[:, 2:3], c.beta_noise, c.beta_quantum)
            beta_dot = self._read(beta_dot, c.beta_dot_noise,
                                  c.beta_quantum / self.step_dt)
            tau = self._read(tau, c.hitch_torque_noise, 0.0)
            base = (base + self.bias[:, 3:5]
                    + torch.randn_like(base) * c.localisation_xy_noise)
            yaw = wrap_to_pi(yaw + self.bias[:, 5]
                             + torch.randn_like(yaw) * c.localisation_yaw_noise)
        if c.true_cart_obs:
            pos, cart_yaw = self._cart_pose_true()
        else:
            pos, cart_yaw = self._predict_cart(base, yaw, beta[:, 0])
        dx, dy, dtheta = self._goal_error(pos, cart_yaw)
        goal = torch.stack((dx, dy, torch.cos(dtheta), torch.sin(dtheta)), dim=1)
        # The two channels the robot has no sensor for, when they are being given away.
        odom = (v, w) if c.observable_only else (v, d.root_lin_vel_b[:, 1:2], w)
        bar = () if c.observable_only else (d.joint_pos[:, self._bar:self._bar + 1],)
        extra = ((self.beta_target.unsqueeze(1),) if c.active_hitch else ())
        return torch.cat(odom + (beta, beta_dot, tau) + bar + (goal,) + extra
                         + (self.actions,), dim=1)

    def _get_observations(self):
        frame = self._frame()
        self.obs_history = torch.roll(self.obs_history, shifts=-1, dims=1)
        self.obs_history[:, -1] = frame
        policy = self.obs_history.reshape(self.num_envs, -1)
        pos, yaw = self._cart_pose_true()
        dx, dy, dtheta = self._goal_error(pos, yaw)
        cart_lin = self.robot.data.body_lin_vel_w[:, self._cart, :2]
        cart_ang = self.robot.data.body_ang_vel_w[:, self._cart, 2:3]
        privileged = torch.cat(
            (torch.stack((dx, dy, torch.cos(dtheta), torch.sin(dtheta)), dim=1),
             cart_lin, cart_ang, self.params)
            + ((self.bias,) if self.cfg.sensor_noise else ()), dim=1)
        return {'policy': policy, 'critic': torch.cat((policy, privileged), dim=1)}

    # ---- reward ----------------------------------------------------------------------
    def _get_rewards(self):
        pos, yaw = self._cart_pose_true()
        dx, dy, dtheta = self._goal_error(pos, yaw)
        distance = torch.sqrt(dx * dx + dy * dy)
        beta = self.robot.data.joint_pos[:, self._yaw]

        progress = self.prev_distance - distance
        self.prev_distance = distance
        near = torch.exp(-(distance / 0.5) ** 2)
        heading = near * torch.exp(-(dtheta / 0.5) ** 2)
        reached = self._reached(distance, dtheta)
        over = torch.clamp(beta.abs() - self.cfg.beta_soft_limit, min=0.0)
        effort = (self.hitch_torque / self.cfg.hitch_torque).abs()
        rate = torch.sum((self.actions - self.prev_actions) ** 2, dim=1)
        jackknifed = beta.abs() > self.cfg.jackknife

        reward = (self.cfg.w_progress * progress
                  + self.cfg.w_near * near * self.step_dt
                  + self.cfg.w_heading * heading * self.step_dt
                  + self.cfg.w_success * reached.float()
                  + self.cfg.w_beta * over ** 2 * self.step_dt
                  + self.cfg.w_effort * effort * self.step_dt
                  + self.cfg.w_action_rate * rate
                  + self.cfg.w_jackknife * jackknifed.float())
        if self.cfg.bay:
            # Normalised by the margin so the weight is read in the units of the other
            # terms rather than in metres, and squared so that the gradient grows as the
            # deck closes instead of being a step at the edge of the margin.
            near_wall = torch.clamp(1.0 - self.wall_gap / self.cfg.bay_wall_margin,
                                    min=0.0)
            reward = (reward
                      + self.cfg.w_wall * near_wall ** 2 * self.step_dt
                      + self.cfg.w_contact * self.last_contact.float())
        return reward

    def _reached(self, distance, dtheta):
        return (distance < self.cfg.goal_position_tolerance) & \
            (dtheta.abs() < self.cfg.goal_heading_tolerance)

    # ---- termination -----------------------------------------------------------------
    def _get_dones(self):
        pos, yaw = self._cart_pose_true()
        dx, dy, dtheta = self._goal_error(pos, yaw)
        distance = torch.sqrt(dx * dx + dy * dy)
        beta = self.robot.data.joint_pos[:, self._yaw]
        # The cart is on its side, or the robot has left the ground: either way the
        # episode is over and the numbers from it mean nothing.
        up = self.robot.data.body_link_quat_w[:, self._cart]
        tipped = 1.0 - 2.0 * (up[:, 1] ** 2 + up[:, 2] ** 2) < 0.7
        self.last_distance = distance
        self.last_yaw_error = dtheta
        self.last_reached = self._reached(distance, dtheta)
        self.last_jackknife = beta.abs() > self.cfg.jackknife
        # Terminal hitch angle, read here because the reset that follows a done inside
        # the same step overwrites it. The figure needs the angle the episode ended on,
        # which for a jackknife is the one that ended it. Cloned, not referenced: beta
        # is a view into joint_pos, so a plain assignment tracks the reset instead.
        self.last_beta = beta.clone()
        if self.cfg.bay:
            self._track_clearance(pos, yaw)
            self.last_contact = self.wall_gap < self.cfg.bay_contact_clearance
            # Contact wins over the goal on the step both happen: the deck is against a
            # wall, and an episode that ends that way did not place the cart.
            self.last_reached = self.last_reached & ~self.last_contact
        done = self.last_jackknife | tipped | self.last_reached | self.last_contact
        timeout = self.episode_length_buf >= self.max_episode_length - 1
        self.last_tipped = tipped
        # A timeout is only the cause of an episode's end if nothing else ended it on
        # the same step, so the two are made exclusive here rather than in the reader.
        self.last_timeout = timeout & ~done
        return done, timeout

    # ---- reset -----------------------------------------------------------------------
    def _reset_idx(self, env_ids):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)
        n, dev = len(env_ids), self.device

        if self.cfg.scenario_seed is None:
            def uniform(low, high, slot=None):
                return torch.rand(n, device=dev) * (high - low) + low
        else:
            draws = self._scenario_draws(env_ids)

            def uniform(low, high, slot=None):
                return draws[:, slot] * (high - low) + low

        # Read the curriculum position once: the free-space goal heading and the bay
        # approach both open out with it.
        progress = (self.cfg.curriculum if self.cfg.curriculum is not None else
                    min(1.0, self.common_step_counter
                        / max(1, self.cfg.curriculum_steps)))

        # Pose: the robot straight along +x at the environment origin, the cart behind it.
        root = self.robot.data.default_root_state[env_ids].clone()
        start_yaw = torch.zeros(n, device=dev)
        if self.cfg.bay:
            # Out in front of the mouth, facing away from it, so the cart lies between
            # the robot and the slot and the only way in is backwards. The approach
            # opens from square-on - a straight reverse - to anything over the
            # curriculum.
            c = self.cfg
            bay_progress = (c.curriculum if c.curriculum is not None else
                            min(1.0, self.common_step_counter
                                / max(1, c.bay_curriculum_steps)))
            spread = (c.start_spread_start
                      + bay_progress * (c.start_spread_end - c.start_spread_start))
            d = uniform(*c.start_distance_range, slot=0)
            phi = uniform(-spread, spread, slot=1)
            start_yaw = wrap_to_pi(phi + uniform(-spread, spread, slot=2))
            root[:, 0] = d * torch.cos(phi)
            root[:, 1] = d * torch.sin(phi)
            # Root state is pos(3), quat(4) as w, x, y, z, then the velocities.
            root[:, 3] = torch.cos(0.5 * start_yaw)
            root[:, 4:6] = 0.0
            root[:, 6] = torch.sin(0.5 * start_yaw)
            self.min_clearance[env_ids] = 10.0
            self.wall_gap[env_ids] = 10.0
            self.entered[env_ids] = False
            # `last_contact` is deliberately NOT cleared here. This runs before the
            # episode metrics further down read it, so clearing it reported a contact
            # rate of zero while the evaluation of the same policy measured 79.7%. It
            # needs no reset: _get_dones recomputes it for every environment each step,
            # the same reason last_reached and last_jackknife are not reset either.
        start_xy = root[:, :2].clone()      # environment frame, before the origin shift
        root[:, :3] += self.scene.env_origins[env_ids]
        self.robot.write_root_pose_to_sim(root[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(torch.zeros_like(root[:, 7:]), env_ids)
        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        joint_pos[:, self._yaw] = (torch.zeros(n, device=dev) if self.cfg.lock_hitch
                                   else uniform(*self.cfg.beta_init_range, slot=3))
        if self._swivels and self.cfg.caster_init_range[1] > 0.0:
            lo, hi = self.cfg.caster_init_range
            for k, j in enumerate(self._swivels):
                joint_pos[:, j] = (uniform(-hi, hi, slot=4 + k) if lo == 0.0
                                   else uniform(lo, hi, slot=4 + k))
        # Start the pitch servo at its towing target rather than at zero. Starting at
        # zero and commanding 0.2 rad asks for Kp * 0.2 = 10 Nm on the first control
        # step, which is the servo's whole torque and kicks the cart off the ground.
        joint_pos[:, self._pitch] = self.cfg.pitch_target
        # The bar turns freely in the hook, so it has to start turned back by as much as
        # the hook is raised - otherwise the cart starts rigidly tilted nose-up, its
        # underside a hand's width below the floor, and PhysX launches it on the first
        # step of every episode.
        joint_pos[:, self._bar] = -self.cfg.pitch_target
        joint_vel = torch.zeros_like(joint_pos)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

        # Goal for the cart.
        beta0 = joint_pos[:, self._yaw]
        if self.cfg.bay:
            # The slot's own pose: centred in it, nose pointing out of the mouth. There
            # is nothing to sample - the bay is where the bay is, and the walls rather
            # than the tolerance decide how precisely it has to be hit.
            cart0, _ = self._predict_cart(start_xy, start_yaw, beta0)
            self.goal[env_ids, 0] = -0.5 * self.cfg.bay_depth
            self.goal[env_ids, 1] = 0.0
            self.goal[env_ids, 2] = 0.0
            distance = torch.linalg.vector_norm(self.goal[env_ids, :2] - cart0, dim=1)
        else:
            # The heading opens out from the bearing to the goal as training proceeds,
            # so early episodes are drive-throughs and later ones include backing into
            # a pose.
            spread = (self.cfg.heading_range_start
                      + progress * (math.pi - self.cfg.heading_range_start))
            distance = uniform(*self.cfg.goal_distance_range, slot=0)
            bearing = uniform(-math.pi, math.pi, slot=1)
            # Where the cart starts, from the reset pose: the robot along +x at the
            # environment origin with the hitch at beta0, so this is exact.
            cart0_x = YAW_AXIS_X + CART_LEVER * torch.cos(beta0)
            cart0_y = CART_LEVER * torch.sin(beta0)
            self.goal[env_ids, 0] = cart0_x + distance * torch.cos(bearing)
            self.goal[env_ids, 1] = cart0_y + distance * torch.sin(bearing)
            self.goal[env_ids, 2] = wrap_to_pi(bearing
                                               + uniform(-spread, spread, slot=2))

        # Payload and actuators.
        payload = uniform(*self.cfg.payload_mass_range, slot=12)
        px = uniform(*self.cfg.payload_x_range, slot=13)
        py = uniform(*self.cfg.payload_y_range, slot=14)
        pz = uniform(*self.cfg.payload_z_range, slot=15)
        self.apply_cart(env_ids, payload, px, py, pz,
                        uniform(*self.cfg.cart_rolling_resistance_range, slot=16),
                        uniform(*self.cfg.swivel_scrub_range, slot=17),
                        uniform(*self.cfg.hitch_effort_range, slot=18),
                        uniform(*self.cfg.wheel_effort_range, slot=19),
                        uniform(*self.cfg.floor_friction_range, slot=20))

        # Success and jackknife rates over the episodes that just ended, as a running
        # average, so the training log says whether the policy is doing the task rather
        # than only what its return is.
        decay = 0.99 ** n
        self.success_rate = decay * self.success_rate + (1 - decay) * \
            float(self.last_reached[env_ids].float().mean())
        self.jackknife_rate = decay * self.jackknife_rate + (1 - decay) * \
            float(self.last_jackknife[env_ids].float().mean())
        self.extras.setdefault('log', {}).update({
            'Episode/success_rate': self.success_rate,
            'Episode/jackknife_rate': self.jackknife_rate})
        if self.cfg.bay:
            # The bay's third outcome, logged during training and not only at evaluation
            # time, because it is the one the task was rewritten to make visible.
            self.contact_rate = decay * self.contact_rate + (1 - decay) * \
                float(self.last_contact[env_ids].float().mean())
            self.extras['log']['Episode/contact_rate'] = self.contact_rate

        # This episode's sensor calibration and actuation delay. The scale errors are
        # multiplicative, so they sit at 1.0 when the noise model is off.
        if self.cfg.sensor_noise:
            c = self.cfg
            self.bias[env_ids, 0] = uniform(*c.odom_v_scale_range, slot=21)
            self.bias[env_ids, 1] = uniform(*c.odom_w_scale_range, slot=22)
            self.bias[env_ids, 2] = uniform(*c.beta_offset_range, slot=23)
            self.bias[env_ids, 3] = uniform(-c.localisation_xy_bias,
                                            c.localisation_xy_bias, slot=24)
            self.bias[env_ids, 4] = uniform(-c.localisation_xy_bias,
                                            c.localisation_xy_bias, slot=25)
            self.bias[env_ids, 5] = uniform(-c.localisation_yaw_bias,
                                            c.localisation_yaw_bias, slot=26)
            low, high = c.latency_steps_range
            if self.cfg.scenario_seed is None:
                self.latency[env_ids] = torch.randint(int(low), int(high) + 1, (n,),
                                                      device=dev)
            else:
                # Half-open on the top end, so the integer draw is uniform over the
                # inclusive range the random path gives.
                self.latency[env_ids] = uniform(float(low), float(high) + 1.0,
                                                slot=27).floor().long().clamp(int(low),
                                                                              int(high))
            self.bias[env_ids, 6] = self.latency[env_ids].float()
        self._queue[env_ids] = 0.0
        self.applied[env_ids] = 0.0
        self.actions[env_ids] = 0.0
        self.prev_actions[env_ids] = 0.0
        self.beta_target[env_ids] = joint_pos[:, self._yaw]
        self.obs_history[env_ids] = 0.0
        # The pose buffers still hold the previous episode until the next physics step,
        # so seed the progress term from the sampled geometry instead of reading them:
        # the cart starts `distance` from its goal by construction.
        self.prev_distance[env_ids] = distance

    def _scenario_draws(self, env_ids):
        """The fixed random numbers for the episodes these environments are starting.

        Drawn on the CPU from a generator per scenario, which costs one seeding and one
        draw of 32 numbers per episode and buys a scenario that is the same under every
        condition. The id is recorded so that a report can be paired on it rather than
        on an episode's position in the log, which is not the same thing: environments
        that terminate early get through more episodes than the ones that do not.
        """
        ids = self.episode_index[env_ids] * self.num_envs + env_ids.to(torch.long)
        rows = torch.empty(len(env_ids), SCENARIO_SLOTS)
        gen = torch.Generator()
        for k, m in enumerate(ids.tolist()):
            gen.manual_seed(int(self.cfg.scenario_seed) * 1_000_003 + int(m))
            rows[k] = torch.rand(SCENARIO_SLOTS, generator=gen)
        self.scenario_id[env_ids] = ids
        self.episode_index[env_ids] += 1
        return rows.to(self.device)

    def apply_cart(self, env_ids, payload, px, py, pz, rolling, scrub, hitch,
                   wheel, friction):
        """Set this episode's machine: payload, drag, servo and wheel limits, floor.

        Separate from ``_reset_idx`` because the actuator sweep in ``authority.py`` sets
        the same quantities on a grid rather than at random, and the two must not drift
        apart - a sweep that changed the mass but left the drag where the random reset
        put it would measure nothing.
        """
        self.params[env_ids] = torch.stack(
            (payload / 100.0, px, py, pz, hitch / 10.6, rolling / 0.03, scrub,
             wheel / 30.0, friction), dim=1)
        self.hitch_limit[env_ids] = hitch
        self._write_payload(env_ids, payload, px, py, pz)
        self.robot.write_joint_effort_limit_to_sim(
            hitch.unsqueeze(1), joint_ids=[self._yaw], env_ids=env_ids)
        self.robot.write_joint_effort_limit_to_sim(
            wheel.unsqueeze(1).repeat(1, len(self._wheels)), joint_ids=self._wheels,
            env_ids=env_ids)
        ids = env_ids.cpu()
        shapes = torch.tensor(self._tyre_shapes)
        static = friction.cpu()
        self._materials[ids.unsqueeze(1), shapes.unsqueeze(0), 0] = static.unsqueeze(1)
        self._materials[ids.unsqueeze(1), shapes.unsqueeze(0), 1] = \
            (0.9 * static).unsqueeze(1)
        self.robot.root_physx_view.set_material_properties(self._materials, ids)
        # Joint friction here is a resisting EFFORT, not a dimensionless coefficient, so
        # the load dependence has to be supplied rather than left to the solver.
        #
        # Isaac Lab switches model on the Isaac Sim major version. Below 5.0 it writes a
        # single coefficient and PhysX bounds the resisting torque by that coefficient
        # times the spatial force the joint transmits, which carries the load scaling
        # for free. From 5.0 it writes static, dynamic and viscous friction, and the
        # static term is the maximum effort that holds the joint at rest - newton metres
        # on a revolute axis, with no reference to the transmitted force. This install
        # is Isaac Sim 5.1.0, so the second model is the one in force.
        #
        # Measured rather than read: with mu_s = 0.5 written to the hitch yaw joint and
        # the servo off, the joint breaks away at an applied 0.5 N.m with an empty cart
        # and at the same 0.5 N.m with 100 kg on it. The threshold does not move with
        # the load, so the figure is a torque.
        #
        # Writing the dimensionless coefficient into that slot, as this file did before,
        # leaves a caster swivel with 0.005-0.012 N.m and a cart wheel with
        # 0.0006-0.002 N.m, which is no resistance at all.
        weight = (asset.CART_BASE_MASS + payload) * 9.81
        # Load per contact. The tow bar carries a share of the cart's weight that this
        # ignores, which makes the friction slightly generous rather than flattering;
        # the contacts are otherwise symmetric about the deck.
        axle_tau = rolling * (weight / max(1, len(self._axles))) \
            * asset.CART_WHEEL_RADIUS
        # Rolling resistance does not fall away once the wheel turns, so the dynamic
        # term equals the static one. A loaded caster breaks away harder than it scrubs,
        # the usual ordering for a swivel, so there the dynamic term is lower.
        axle_cols = axle_tau.unsqueeze(1).repeat(1, len(self._axles))
        self.robot.write_joint_friction_coefficient_to_sim(
            axle_cols, joint_dynamic_friction_coeff=axle_cols,
            joint_ids=self._axles, env_ids=env_ids)
        if self._swivels:
            swivel_tau = scrub * (weight / len(self._swivels)) * self.cfg.scrub_radius
            cols = swivel_tau.unsqueeze(1).repeat(1, len(self._swivels))
            self.robot.write_joint_friction_coefficient_to_sim(
                cols, joint_dynamic_friction_coeff=0.8 * cols,
                joint_ids=self._swivels, env_ids=env_ids)

    def _write_payload(self, env_ids, mass, px, py, pz):
        """Add the payload to the cart body: mass, centre of mass and inertia together.

        Setting the mass without the inertia is the usual way to get a cart that weighs
        80 kg and turns like an empty one, so all three move at once. The payload is
        treated as a box of its own and the parallel-axis terms are added about the
        combined centre of mass.
        """
        ids = env_ids.cpu()
        cart = self._cart
        mass = mass.cpu()
        p = torch.stack((px, py, pz), dim=1).cpu()
        empty = self._default_masses[ids, cart]
        total = empty + mass
        empty_com = self._default_coms[ids, cart, :3]
        com = (empty_com * empty.unsqueeze(1) + p * mass.unsqueeze(1)) / total.unsqueeze(1)

        def box_inertia(m, box):
            sx, sy, sz = box
            return torch.stack((m * (sy * sy + sz * sz) / 12.0,
                                m * (sx * sx + sz * sz) / 12.0,
                                m * (sx * sx + sy * sy) / 12.0), dim=1)

        def shift(diag, m, offset):
            x, y, z = offset[:, 0], offset[:, 1], offset[:, 2]
            return torch.stack((diag[:, 0] + m * (y * y + z * z),
                                diag[:, 1] + m * (x * x + z * z),
                                diag[:, 2] + m * (x * x + y * y)), dim=1)

        total_i = shift(box_inertia(empty, asset.CART_BOX), empty, empty_com - com) \
            + shift(box_inertia(mass, asset.PAYLOAD_BOX), mass, p - com)

        self._masses[ids, cart] = total
        self._coms[ids, cart, :3] = com
        self._inertias[ids, cart, 0] = total_i[:, 0]
        self._inertias[ids, cart, 4] = total_i[:, 1]
        self._inertias[ids, cart, 8] = total_i[:, 2]
        view = self.robot.root_physx_view
        view.set_masses(self._masses, ids)
        view.set_coms(self._coms, ids)
        view.set_inertias(self._inertias, ids)
