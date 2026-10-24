"""Arguments shared by train.py and play.py, and the cfg they build.

Kept out of the package because it has to be importable before the Omniverse app is
launched: the parser is built, the app comes up, and only then can anything touch
``amr_lab.towing_env``.
"""

import argparse


def add_arguments(parser: argparse.ArgumentParser):
    parser.add_argument('--num-envs', type=int, default=None)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--max-iterations', type=int, default=None)
    parser.add_argument('--cart-wheels', choices=('trailer', 'caster4'), default='trailer',
                        help='single-axle trailer, or the four-caster cart the real one is')
    parser.add_argument('--passive-hitch', action='store_true',
                        help='limp hitch yaw servo: the baseline the campaign measured')
    parser.add_argument('--true-cart-obs', action='store_true',
                        help='give the actor the true cart pose instead of the one '
                             'predicted from the hitch angle (the observability ablation)')
    parser.add_argument('--payload', type=float, nargs=2, default=None,
                        metavar=('MIN', 'MAX'),
                        help='payload mass range on the deck [kg], on top of 22.6 kg')
    parser.add_argument('--hitch-torque', type=float, nargs=2, default=None,
                        metavar=('MIN', 'MAX'),
                        help='hitch servo effort limit range [Nm]; 10.6 is the XM540 '
                             'that is fitted, a larger range stands for a larger servo')
    parser.add_argument('--pitch-torque', type=float, default=None, metavar='NM',
                        help='effort limit of the hook pitch servo. It follows '
                             '--hitch-torque when unset, which confounds a yaw sizing '
                             'sweep with the tail-press of the pitch servo; pin it to '
                             '10.6 to sweep the yaw limit alone')
    parser.add_argument('--floor-friction', type=float, nargs=2, default=None,
                        metavar=('MIN', 'MAX'),
                        help='tyre-floor static friction range. Keep the bottom at or '
                             'above 0.7: authority.py shows the robot cannot hold speed '
                             'below that even with an empty cart, so a wider range '
                             'trains the policy on episodes it cannot complete')
    parser.add_argument('--hitch-dc', action='store_true',
                        help='model the hitch servo with a DC motor torque-speed curve '
                             'instead of a flat stall-torque limit')
    parser.add_argument('--physics-hz', type=float, default=None,
                        help='physics rate; the control rate is held at 20 Hz by '
                             'recomputing the decimation, so this changes the '
                             'integrator and nothing the policy sees')
    parser.add_argument('--solver-iters', type=int, nargs=2, default=None,
                        metavar=('POS', 'VEL'),
                        help='PhysX position and velocity solver iteration counts')
    parser.add_argument('--lock-hitch', action='store_true',
                        help='hold the hitch yaw at zero with the servo instead of '
                             'commanding it; the third mechanical condition between '
                             'driven and limp')
    parser.add_argument('--caster-init', type=float, default=None,
                        help='half-width in rad of the random initial swivel angle of '
                             'each cart caster (default 0: all four start straight)')
    parser.add_argument('--heading-spread', type=float, default=None,
                        help='half-width in rad of the goal heading window about the '
                             'bearing to the goal; pi evaluates full-pose reaching')
    parser.add_argument('--goal-tolerance', type=float, nargs=2, default=None,
                        metavar=('METRES', 'RADIANS'),
                        help='how close counts as reached. The default matches what E5 '
                             'measured (250 mm, 0.35 rad); tightening it below the '
                             'baseline leaves the policy with no success to learn from')
    parser.add_argument('--episode-length', type=float, default=None)
    parser.add_argument('--control-hz', type=float, default=None,
                        help='policy rate. The default 20 Hz is what amr_base runs the '
                             'wheel driver and odometry at; a checkpoint has to be '
                             'played at the rate it was trained on')
    parser.add_argument('--legacy-obs', action='store_true',
                        help='also give the actor the lateral velocity and the bar '
                             'angle, which the real robot cannot measure (ablation, and '
                             'what checkpoints from before this flag were trained on)')
    parser.add_argument('--no-sensor-noise', dest='sensor_noise', action='store_false',
                        help='noiseless, perfectly calibrated, zero-latency sensors')
    parser.add_argument('--latency', type=int, nargs=2, default=None,
                        metavar=('MIN', 'MAX'),
                        help='actuation delay range in control steps')
    parser.add_argument('--perfect-sensors', action='store_true',
                        help='keep the observation shape but take the noise, the '
                             'calibration error, the quantisation and the delay out of '
                             'it. This is how to ask a noise-trained checkpoint what the '
                             'sensor model cost it; --no-sensor-noise changes the '
                             'critic shape and will not load one')
    parser.add_argument('--pitch-target', type=float, default=None,
                        help='hitch pitch angle held while towing [rad]. 0 is the '
                             'seated bar; 0.2 is the catching angle and saturates the '
                             'servo, which pushes the robot tail down')
    parser.add_argument('--rigid-lock', action='store_true',
                        help='pin the hitch joint mechanically instead of commanding a '
                             'servo to zero; implies --passive-hitch --lock-hitch')
    parser.add_argument('--wheel-torque', type=float, nargs=2, default=None,
                        help='per-episode wheel effort limit range in Nm; the default '
                             '11-30 spans the MDH300 rating to a peak')
    parser.add_argument('--caster-clearance', type=float, default=None,
                        help='support-caster clearance in m (default 0.010); builds a '
                             'separate asset, for the sensitivity check')
    parser.add_argument('--rear-caster-clearance', type=float, default=None,
                        help='clearance of the REAR support caster alone in m; the real '
                             'build has the front caster about 10 mm up and the rear '
                             'one flush, so 0 is that build. Front and rear flush '
                             'together would be the indeterminate four-point case, '
                             'which measures the solver rather than the robot')
    parser.add_argument('--bay', action='store_true',
                        help='back the cart into a loading bay between two walls '
                             'instead of to a pose on an empty floor')
    parser.add_argument('--bay-width', type=float, default=None,
                        help='clear span of the bay in m (deck is 0.56); default 0.80')
    parser.add_argument('--curriculum', type=float, default=None,
                        help='pin the goal-heading and bay-approach curricula to this '
                             'fraction of their schedule instead of reading the step '
                             'counter, which is zero in a fresh process: an evaluation '
                             'without it samples the easy end of the schedule the '
                             'policy trained on. Use 1.0 to evaluate on the '
                             'distribution training ended with')
    parser.add_argument('--scenario-seed', type=int, default=None,
                        help='draw each episode from a generator seeded by the '
                             'episode index, so that every condition sees the same '
                             'scenarios and can be compared pair by pair')
    parser.add_argument('--run-name', type=str, default='')
    parser.add_argument('--rebuild-asset', action='store_true')
    parser.add_argument('--visual-meshes', action='store_true',
                        help='attach the CAD meshes to the asset, for a figure or a '
                             'recording; not for training')
    parser.add_argument('--backdrop', action='store_true',
                        help='put the scene in the warehouse the ROS simulator uses, '
                             'for a recording; visual only, and not for training')
    return parser


def build_cfg(args):
    """Build the env cfg. Only call this once the app is up."""
    from amr_lab import asset
    from amr_lab.towing_env import make_robot_cfg, TowingEnvCfg

    cfg = TowingEnvCfg()
    cfg.cart_wheels = args.cart_wheels
    cfg.active_hitch = not args.passive_hitch
    cfg.true_cart_obs = args.true_cart_obs
    if args.payload is not None:
        cfg.payload_mass_range = tuple(args.payload)
    if args.hitch_torque is not None:
        cfg.hitch_effort_range = tuple(args.hitch_torque)
        # The servo cannot be given a limit above what it is configured with, so raise
        # the configured one to the top of the sweep and let the per-episode limit bite.
        cfg.hitch_torque = max(cfg.hitch_torque, float(args.hitch_torque[1]))
    if args.pitch_torque is not None:
        cfg.pitch_torque = args.pitch_torque
    if args.goal_tolerance is not None:
        cfg.goal_position_tolerance, cfg.goal_heading_tolerance = args.goal_tolerance
    if args.floor_friction is not None:
        cfg.floor_friction_range = tuple(args.floor_friction)
    cfg.bay = getattr(args, 'bay', False)
    if getattr(args, 'bay_width', None) is not None:
        cfg.bay_width = args.bay_width
    if cfg.bay:
        # Backing into a slot from 3.5 m out takes longer than crossing to a free-space
        # pose, and an episode that times out mid-manoeuvre is scored as a failure that
        # the controller did not commit. 15 s was enough for the free floor; this is not
        # a tuning knob, it is the time the manoeuvre takes.
        cfg.episode_length_s = 20.0
    if args.episode_length is not None:
        cfg.episode_length_s = args.episode_length
    if args.control_hz is not None:
        cfg.decimation = max(1, round(1.0 / (cfg.sim.dt * args.control_hz)))
    cfg.lock_hitch = args.lock_hitch
    if getattr(args, 'rigid_lock', False):
        # The mechanical condition: the hitch is not commanded and nothing can move it.
        cfg.rigid_lock = True
        cfg.lock_hitch = True
        cfg.active_hitch = False
    if getattr(args, 'wheel_torque', None) is not None:
        cfg.wheel_effort_range = tuple(args.wheel_torque)
        cfg.wheel_torque = max(cfg.wheel_torque, float(args.wheel_torque[1]))
    cfg.backdrop = getattr(args, 'backdrop', False)
    cfg.hitch_dc = args.hitch_dc
    if args.solver_iters is not None:
        cfg.solver_position_iterations, cfg.solver_velocity_iterations = args.solver_iters
    if args.physics_hz is not None:
        cfg.sim.dt = 1.0 / args.physics_hz
        control_hz = args.control_hz if args.control_hz else 20.0
        cfg.decimation = max(1, round(args.physics_hz / control_hz))
    if args.caster_init is not None:
        cfg.caster_init_range = (0.0, args.caster_init)
    if args.heading_spread is not None:
        # The window opens from heading_range_start towards pi over curriculum_steps.
        # Setting the start is therefore a floor on the window, which is what an
        # out-of-distribution evaluation wants: it holds for the whole run rather than
        # depending on how far a curriculum happened to advance.
        cfg.heading_range_start = args.heading_spread
    cfg.observable_only = not args.legacy_obs
    cfg.sensor_noise = args.sensor_noise
    if args.latency is not None:
        cfg.latency_steps_range = tuple(args.latency)
    if args.perfect_sensors:
        cfg.odom_v_noise = cfg.odom_w_noise = 0.0
        cfg.odom_v_quantum = cfg.odom_w_quantum = 0.0
        cfg.odom_v_scale_range = cfg.odom_w_scale_range = (1.0, 1.0)
        cfg.beta_noise = cfg.beta_dot_noise = cfg.beta_quantum = 0.0
        cfg.beta_offset_range = (0.0, 0.0)
        cfg.hitch_torque_noise = 0.0
        cfg.localisation_xy_noise = cfg.localisation_yaw_noise = 0.0
        cfg.localisation_xy_bias = cfg.localisation_yaw_bias = 0.0
        cfg.latency_steps_range = (0, 0)
    if args.pitch_target is not None:
        cfg.pitch_target = args.pitch_target
    cfg.curriculum = getattr(args, 'curriculum', None)
    cfg.scenario_seed = getattr(args, 'scenario_seed', None)
    if args.num_envs is not None:
        cfg.scene.num_envs = args.num_envs
    if args.seed is not None:
        cfg.seed = args.seed
    cfg.__post_init__()
    path = asset.ensure(cfg.cart_wheels, rebuild=args.rebuild_asset,
                        visuals=getattr(args, 'visual_meshes', False),
                        clearance=getattr(args, 'caster_clearance', None),
                        rear_clearance=getattr(args, 'rear_caster_clearance', None))
    cfg.robot = make_robot_cfg(cfg, path)
    return cfg, path
