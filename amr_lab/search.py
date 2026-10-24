"""Search for an input sequence that does the task, with the simulator as the model.

The question this answers is the one a learned policy cannot: when a mechanical
condition fails at the task, is that the mechanism's limit or the learner's? A policy
that fails is open to the reading that the learner, not the hitch, is what failed.

So this removes the learner. It plans open loop with the cross-entropy method, using the
simulator itself as its model: every candidate is a full command sequence rolled from the
same initial state to the end of the episode, the best are kept, a distribution is refitted
to them, and the search repeats. The model error is therefore zero --- it plans in the
dynamics it is scored in --- and the compute per episode is thousands of times what a 20 Hz
controller could spend, so the result is an upper bound on what any controller of this
command set could achieve, not an estimate of what one would.

Read it that way. If the search cannot reverse the cart, no controller can. If it can at
some rate, that rate is what a passive hitch leaves on the table and the honest claim
becomes a comparison of achievable rates rather than of possibility.

  python amr_lab/search.py --headless --cart-wheels caster4 --passive-hitch \
      --scenarios 40 --cem-iters 6

Every candidate in a batch shares one scenario: the same start pose, goal, payload,
friction, hitch limit and caster drag, written over every environment after the reset, so
2048 environments are 2048 candidate command sequences for one episode rather than 2048
different episodes. Sensor noise and command latency are off: the question is what the
mechanism admits, and a search that had to plan through noise it cannot observe would
answer a different one.
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cli  # noqa: E402, I100

parser = argparse.ArgumentParser(description=__doc__)
cli.add_arguments(parser)
parser.add_argument('--scenarios', type=int, default=40,
                    help='episodes to search, each from its own sampled scenario')
parser.add_argument('--cem-iters', type=int, default=6, help='CEM refits per scenario')
parser.add_argument('--elite-frac', type=float, default=0.05)
parser.add_argument('--segment', type=int, default=20,
                    help='control steps each command is held for; the search is over '
                         'piecewise-constant commands, which is what keeps the '
                         'dimension small enough for CEM to make progress')
parser.add_argument('--backward-only', action='store_true', default=True,
                    help='search only scenarios whose goal is behind the cart')
parser.add_argument('--init-std', type=float, default=0.6)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch  # noqa: E402, I100


def broadcast(env):
    """Give every environment the scenario of environment 0."""
    n, ids = env.num_envs, env.robot._ALL_INDICES
    root = env.robot.data.root_state_w.clone()
    root[:, :3] = root[0:1, :3] - env.scene.env_origins[0:1] + env.scene.env_origins
    root[:, 3:] = root[0:1, 3:]
    env.robot.write_root_pose_to_sim(root[:, :7], ids)
    env.robot.write_root_velocity_to_sim(torch.zeros_like(root[:, 7:]), ids)
    jp = env.robot.data.joint_pos.clone()
    jp[:] = jp[0:1]
    env.robot.write_joint_state_to_sim(jp, torch.zeros_like(jp), env_ids=ids)
    env.goal[:] = env.goal[0:1]
    env.prev_distance[:] = env.prev_distance[0]
    p = env.params[0]
    rep = lambda k, scale=1.0: p[k].repeat(n) * scale  # noqa: E731
    env.apply_cart(ids, rep(0, 100.0), rep(1), rep(2), rep(3),
                   rep(5, 0.03), rep(6), rep(4, 10.6), rep(7, 30.0), rep(8))
    env.beta_target[:] = env.beta_target[0]
    env.obs_history[:] = 0.0
    env.actions[:] = 0.0
    env.prev_actions[:] = 0.0
    env._queue[:] = 0.0
    env.applied[:] = 0.0
    env.episode_length_buf[:] = 0


def rollout(env, seqs, steps, seg):
    """Run one command sequence per environment and score it.

    An environment that finishes is left to reset and is ignored from then on, so the
    score is of the first episode only. The cost tracked is the smallest combined
    position and heading error the cart reached at any point, which is smooth enough for
    the refit while still being minimised by actually arriving.
    """
    n, dev = env.num_envs, env.device
    best = torch.full((n,), 1e3, device=dev)
    done_yet = torch.zeros(n, dtype=torch.bool, device=dev)
    reached = torch.zeros(n, dtype=torch.bool, device=dev)
    with torch.no_grad():
        for t in range(steps):
            env.step(seqs[:, t // seg].clamp(-1.0, 1.0))
            pos, yaw = env._cart_pose_true()
            dx, dy, dth = env._goal_error(pos, yaw)
            err = torch.sqrt(dx * dx + dy * dy) + 0.5 * dth.abs()
            best = torch.where(done_yet, best, torch.minimum(best, err))
            fin = env.reset_terminated | env.reset_time_outs
            new = fin & ~done_yet
            reached |= new & env.last_reached
            done_yet |= fin
            if bool(done_yet.all()):
                break
    return -best + 100.0 * reached.float(), reached


def main():
    from amr_lab.towing_env import TowingEnv

    if args.num_envs is None:
        args.num_envs = 2048
    cfg, _ = cli.build_cfg(args)
    cfg.sensor_noise = False
    cfg.latency_steps_range = (0, 0)
    env = TowingEnv(cfg)
    n, dev = env.num_envs, env.device
    steps = int(env.max_episode_length)
    seg = args.segment
    nseg = (steps + seg - 1) // seg
    dim = cfg.action_space
    elite = max(8, int(args.elite_frac * n))
    print(f'[search] {n} candidates, {steps} steps, {nseg} segments of {seg}, '
          f'{dim} inputs, {elite} elite, {args.cem_iters} refits')
    print(f'[search] cart {cfg.cart_wheels}, hitch '
          f'{"active" if cfg.active_hitch else ("locked" if cfg.lock_hitch else "passive")}')

    solved = 0
    for k in range(args.scenarios):
        while True:
            env.reset()
            if not args.backward_only or bool(env.goal[0, 0] < -0.85):
                break
        broadcast(env)
        start = (env.robot.data.root_state_w.clone(),
                 env.robot.data.joint_pos.clone(), env.goal[0].clone())
        mean = torch.zeros(nseg, dim, device=dev)
        std = torch.full((nseg, dim), args.init_std, device=dev)
        best_score, best_reached = -1e9, False
        for it in range(args.cem_iters):
            seqs = mean + std * torch.randn(n, nseg, dim, device=dev)
            score, reached = rollout(env, seqs, steps, seg)
            top = score.topk(elite).indices
            mean = seqs[top].mean(dim=0)
            std = seqs[top].std(dim=0).clamp(min=0.05)
            if float(score.max()) > best_score:
                best_score, best_reached = float(score.max()), bool(reached[top[0]])
            # put the scenario back for the next refit
            env.reset()
            env.robot.write_root_pose_to_sim(start[0][:, :7], env.robot._ALL_INDICES)
            env.robot.write_root_velocity_to_sim(torch.zeros_like(start[0][:, 7:]),
                                                 env.robot._ALL_INDICES)
            env.robot.write_joint_state_to_sim(start[1], torch.zeros_like(start[1]),
                                               env_ids=env.robot._ALL_INDICES)
            env.goal[:] = start[2]
            broadcast(env)
        solved += int(bool(reached.any()) or best_reached)
        print(f'[search] scenario {k + 1:3d}/{args.scenarios}  '
              f'best cost {-best_score if best_score < 50 else 0:6.3f}  '
              f'reached {"YES" if (bool(reached.any()) or best_reached) else "no "}  '
              f'running {solved}/{k + 1} = {100.0 * solved / (k + 1):5.1f}%', flush=True)

    print(f'\n=== simulator-as-model search, {cfg.cart_wheels} cart, '
          f'{"active" if cfg.active_hitch else ("locked" if cfg.lock_hitch else "passive")}'
          f' hitch, goals behind the cart')
    print(f'  scenarios solved: {solved}/{args.scenarios} '
          f'({100.0 * solved / max(1, args.scenarios):.1f}%)')
    print(f'  budget per scenario: {n * args.cem_iters} full-episode rollouts')
    env.close()


main()
simulation_app.close()
