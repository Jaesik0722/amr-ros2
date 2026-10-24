"""Record frames of a trained policy doing the task, for the paper's figure and video.

The point of the figure is the comparison, so the three mechanical conditions have to be
shown on the SAME episode: same goal, same payload, same floor, same initial hitch angle.
Relying on the seed for that does not work --- the locked condition skips one random draw
(its hitch angle starts at zero rather than being sampled), which shifts every draw after
it --- so a scenario is written to a file by the first run and read back by the others.

  # active, and save the scenario it drew
  python amr_lab/capture.py --headless --enable_cameras --cart-wheels caster4 \
      --checkpoint <active>  --save-scenario /tmp/sc.pt --out /tmp/frames/active
  # the other two on that same scenario
  python amr_lab/capture.py --headless --enable_cameras --cart-wheels caster4 \
      --passive-hitch --checkpoint <passive> --load-scenario /tmp/sc.pt \
      --out /tmp/frames/passive

Frames are written as PNGs with their control step in the name, so a filmstrip can pick
the moments it wants afterwards rather than deciding here.
"""

import argparse
import math
import os
import sys

from isaaclab.app import AppLauncher

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cli  # noqa: E402, I100

parser = argparse.ArgumentParser(description=__doc__)
cli.add_arguments(parser)
parser.add_argument('--checkpoint', type=str, default=None)
parser.add_argument('--out', type=str, required=True, help='directory for the PNGs')
parser.add_argument('--save-scenario', type=str, default=None)
parser.add_argument('--load-scenario', type=str, default=None)
parser.add_argument('--backward', action=argparse.BooleanOptionalAction, default=True,
                    help='only record an episode whose goal is behind the cart. The bay '
                         'task puts its goal at -bay_depth/2, which never satisfies '
                         'this, so --no-backward is required there or the reset loop '
                         'never terminates')
parser.add_argument('--every', type=int, default=4, help='record every Nth control step')
parser.add_argument('--goal-marker', action=argparse.BooleanOptionalAction, default=True,
                    help='draw the commanded cart pose on the floor. Visual only, and '
                         'built only when asked, so no evaluation ever has it in frame')
parser.add_argument('--env-spacing', type=float, default=None,
                    help='metres between environment origins. Only worth setting with '
                         '--num-envs above 1, to push the neighbours out of shot')
parser.add_argument('--resolution', type=int, nargs=2, default=(1280, 720))
parser.add_argument('--eye', type=float, nargs=3, default=(3.2, -3.6, 2.6))
parser.add_argument('--lookat', type=float, nargs=3, default=(-0.4, 0.0, 0.2))
parser.add_argument('--auto-camera', action='store_true',
                    help='place a fixed camera that frames the cart start AND the goal, '
                         'computed from the scenario once it is loaded. A chase camera '
                         'cannot do this: the goal can be 3.5 m away in any direction, '
                         'so a shot tight enough to show the hitch leaves the goal out '
                         'of frame and a viewer cannot tell a success from a timeout')
parser.add_argument('--follow', action='store_true',
                    help='chase camera: --eye and --lookat become offsets from the '
                         'robot rather than world points, so the shot can be close '
                         'enough to see the hitch over a three-metre run')
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import imageio.v2 as imageio  # noqa: E402, I100
import torch  # noqa: E402, I100
from isaaclab_rl.rsl_rl import handle_deprecated_rsl_rl_cfg, RslRlVecEnvWrapper  # noqa: E402, I100
from rsl_rl.runners import OnPolicyRunner  # noqa: E402, I100


def save_scenario(env, path):
    """Record everything a reset drew, so another condition can start where this did.

    The goal, the randomised cart parameters and the initial hitch angle are not the
    whole of it. The four caster swivels are drawn per episode as well, and so is the
    episode's sensor calibration and its command delay; leaving those out would let
    three runs that share a 'scenario' still begin from different castor angles and
    different calibration, which is exactly the comparison the figure claims to make.
    The robot's own root pose needs no saving --- the reset puts it at the environment
    origin along +x every time.
    """
    torch.save({'goal': env.goal[0].cpu(),
                'params': env.params[0].cpu(),
                'beta0': env.robot.data.joint_pos[0, env._yaw].cpu(),
                'joint_pos': env.robot.data.joint_pos[0].cpu(),
                'bias': env.bias[0].cpu(),
                'latency': env.latency[0].cpu()}, path)
    print(f'[capture] scenario written to {path}')


def load_scenario(env, path):
    """Write a saved scenario over whatever the reset drew."""
    sc = torch.load(path)
    ids = env.robot._ALL_INDICES
    env.goal[:] = sc['goal'].to(env.device)
    jp = env.robot.data.joint_pos.clone()
    if 'joint_pos' in sc:
        # Everything the reset drew, the caster swivels among them.
        jp[:] = sc['joint_pos'].to(env.device)
    # a locked hitch holds zero whatever the scenario says; everything else follows it
    if env.cfg.lock_hitch:
        jp[:, env._yaw] = 0.0
    else:
        jp[:, env._yaw] = sc['beta0'].to(env.device)
    env.robot.write_joint_state_to_sim(jp, torch.zeros_like(jp), env_ids=ids)
    if 'bias' in sc:
        env.bias[:] = sc['bias'].to(env.device)
        env.latency[:] = sc['latency'].to(env.device)
    env.beta_target[:] = jp[:, env._yaw]
    p = sc['params'].to(env.device)
    n = env.num_envs
    rep = lambda k, s=1.0: p[k].repeat(n) * s  # noqa: E731
    env.apply_cart(ids, rep(0, 100.0), rep(1), rep(2), rep(3),
                   rep(5, 0.03), rep(6), rep(4, 10.6), rep(7, 30.0), rep(8))
    pos, yaw = env._cart_pose_true()
    dx, dy, _ = env._goal_error(pos, yaw)
    env.prev_distance[:] = torch.sqrt(dx * dx + dy * dy)
    print(f'[capture] scenario read from {path}')


def main():
    import importlib.metadata as metadata

    from amr_lab.agents.rsl_rl_ppo_cfg import TowingPPORunnerCfg
    from amr_lab.towing_env import TowingEnv

    if args.num_envs is None:
        # One environment is the obvious choice for a recording and it is wrong. The
        # outcome of a given scenario depends on how many environments share the scene
        # -- the same checkpoint on the same eight scenarios reaches 0 of 8 at
        # --num-envs 1 and 4 of 8 at --num-envs 8 -- because PhysX solves the whole
        # scene as one batch and this task bifurcates (a jackknife is a fork in the
        # trajectory, not a margin). Recording alone would therefore show a policy
        # folding the cart that the tables say places it. Record at the count the
        # evaluation uses, or at least well above one, and film environment 0.
        args.num_envs = 1
    cfg, _ = cli.build_cfg(args)
    if args.env_spacing is not None:
        cfg.scene.env_spacing = args.env_spacing
    # The figure is recorded under the conditions the paper evaluates, noise and
    # latency included. Turning the noise model off would also drop the seven
    # calibration channels from the critic's input, and these checkpoints were trained
    # with them, so the recording would not load at all.
    cfg.viewer.resolution = tuple(args.resolution)
    cfg.viewer.eye = tuple(args.eye)
    cfg.viewer.lookat = tuple(args.lookat)
    if args.follow:
        # A fixed camera has to hold the whole run in frame, which leaves the robot a
        # tenth of the picture; this tracks the robot instead, as the chase panel of
        # the ROS recordings does.
        cfg.viewer.origin_type = 'asset_root'
        cfg.viewer.asset_name = 'robot'
        cfg.viewer.env_index = 0
    env = TowingEnv(cfg, render_mode='rgb_array')

    policy = None
    if args.checkpoint:
        # The version argument is not optional: rsl-rl-lib 5.0.1 needs it to decide
        # which of the two config layouts it is being handed.
        agent = handle_deprecated_rsl_rl_cfg(TowingPPORunnerCfg(),
                                             metadata.version('rsl-rl-lib'))
        wrapped = RslRlVecEnvWrapper(env, clip_actions=None)
        runner = OnPolicyRunner(wrapped, agent.to_dict(), log_dir=None, device=agent.device)
        runner.load(args.checkpoint)
        policy = runner.get_inference_policy(device=env.device)
        print(f'[capture] policy {args.checkpoint}')

    while True:
        env.reset()
        if not args.backward:
            break
        # The same criterion the evaluation uses for "behind the cart": the bearing
        # from the cart's start to the goal, past a right angle. A threshold on the
        # goal's x alone is a different and much harder subset -- it selects goals
        # that are far behind rather than merely behind -- so a clip chosen that way
        # is not drawn from the class the tables report.
        cart0, _ = env._cart_pose_true()
        bearing = torch.atan2(env.goal[0, 1] - cart0[0, 1],
                              env.goal[0, 0] - cart0[0, 0])
        if bool(bearing.abs() > math.pi / 2):
            break
    if args.load_scenario:
        load_scenario(env, args.load_scenario)
    elif args.save_scenario:
        save_scenario(env, args.save_scenario)
        # And then start from it, rather than from the state it was read out of. A
        # reset lets the sim settle before this point; loading writes the joint state
        # back over that settled state, so a run that saves a scenario and a run that
        # loads the same file would otherwise begin from slightly different contacts
        # and diverge. Reading it straight back makes the two paths identical, which
        # is what lets the figure and the video show the same episode.
        load_scenario(env, args.save_scenario)

    if args.auto_camera:
        # Side-on to the line the cart has to travel, far enough back to hold both ends
        # of it, and higher the longer that line is.
        cart0, _ = env._cart_pose_true()
        sx, sy = float(cart0[0, 0]), float(cart0[0, 1])
        gx, gy = float(env.goal[0, 0]), float(env.goal[0, 1])
        cx, cy = 0.5 * (sx + gx), 0.5 * (sy + gy)
        span = math.hypot(gx - sx, gy - sy) + 2.4
        bearing = math.atan2(gy - sy, gx - sx) - math.pi / 2
        origin = env.scene.env_origins[0]
        env.sim.set_camera_view(
            eye=(cx + span * 0.85 * math.cos(bearing) + float(origin[0]),
                 cy + span * 0.85 * math.sin(bearing) + float(origin[1]),
                 span * 0.62),
            target=(cx + float(origin[0]), cy + float(origin[1]), 0.25))
        print(f'[capture] auto camera: span {span:.2f} m about ({cx:.2f}, {cy:.2f})')

    os.makedirs(args.out, exist_ok=True)
    # Figure 1 is drawn from these, not from the frames: at the scale the camera has to
    # use to keep robot, cart and goal all in shot, beta is a few pixels and the three
    # conditions look alike. The trajectory and beta(t) are where they differ.
    traj = {'goal': env.goal[0, :].cpu().clone(),
            'beta_limit': cfg.jackknife, 'dt': env.step_dt,
            'base': [], 'cart': [], 'beta': [], 'pitch': [], 'z': []}

    def log():
        base_xy, base_yaw = env._base_pose()
        cart_xy, cart_yaw = env._cart_pose_true()
        traj['base'].append(torch.cat((base_xy[0], base_yaw[0:1])).cpu().clone())
        traj['cart'].append(torch.cat((cart_xy[0], cart_yaw[0:1])).cpu().clone())
        traj['beta'].append(float(env.robot.data.joint_pos[0, env._yaw]))
        # Body pitch and height. The support casters are held clear of the floor, so the
        # chassis rocks fore-and-aft through that clearance while driving; a recording
        # that only logs x, y and yaw cannot show it, and at a degree or two it is easier
        # to measure than to see in the render.
        q = env.robot.data.root_quat_w[0]
        sinp = float(torch.clamp(2.0 * (q[0] * q[2] - q[3] * q[1]), -1.0, 1.0))
        traj['pitch'].append(math.asin(sinp))
        traj['z'].append(float(env.robot.data.root_pos_w[0, 2]))

    # rsl-rl 5.0.1 indexes the observation groups itself (obs_groups in the agent cfg),
    # so the inference policy takes the whole dict, not the policy tensor out of it.
    obs = env._get_observations()
    frames = 0
    # The first render of the session comes back black: the RTX pipeline has not
    # resolved a frame yet. Warm it up before anything is written.
    if args.goal_marker:
        env.show_goal()
    for _ in range(4):
        env.render()
    with torch.no_grad():
        for t in range(int(env.max_episode_length)):
            # Both the pose and the frame are taken before the step. A step that
            # terminates also resets inside itself, so anything read afterwards
            # belongs to the next episode -- which is what used to make the last
            # frame of a recording jump.
            log()
            if t % args.every == 0:
                rgb = env.render()
                if rgb is not None:
                    imageio.imwrite(os.path.join(args.out, f'{t:04d}.png'), rgb)
                    frames += 1
            action = (policy(obs) if policy is not None
                      else torch.zeros(env.num_envs, cfg.action_space, device=env.device))
            obs = env.step(action)[0]
            if bool(env.reset_terminated[0]) or bool(env.reset_time_outs[0]):
                traj['beta'].append(float(env.last_beta[0]))
                traj['ended'] = t
                traj['jackknife'] = bool(env.last_jackknife[0])
                traj['reached'] = bool(env.last_reached[0])
                # A clip that ended early and neither reached nor folded has a third
                # cause, and a caption that calls it a timeout is simply wrong: in the
                # bay the usual one is the deck touching a wall. Report the cause rather
                # than leaving it to be inferred from the step count.
                traj['contact'] = bool(env.last_contact[0]) and bool(env.cfg.bay)
                traj['tipped'] = bool(env.last_tipped[0])
                print(f'[capture] episode ended at step {t}, '
                      f'reached={bool(env.last_reached[0])}, '
                      f'jackknife={bool(env.last_jackknife[0])}, '
                      f'contact={traj["contact"]}, tipped={traj["tipped"]}, '
                      f'terminal |beta| {abs(float(env.last_beta[0])):.2f} rad')
                break
    for k in ('base', 'cart'):
        traj[k] = torch.stack(traj[k])
    # The terminal angle appended above has no pose beside it, so beta can be one
    # longer than the poses. The plot reads them by their own lengths.
    traj['beta'] = torch.tensor(traj['beta'])
    for k in ('pitch', 'z'):
        traj[k] = torch.tensor(traj[k])
    torch.save(traj, os.path.join(args.out, 'traj.pt'))
    print(f'[capture] {frames} frames and {len(traj["beta"])} poses in {args.out}')
    env.close()


main()
simulation_app.close()
