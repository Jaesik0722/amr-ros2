"""Roll a policy out over a fixed set of scenarios and write one row per episode.

  python amr_lab/eval_pairs.py --headless --bay --cart-wheels caster4 \
      --curriculum 1.0 --scenario-seed 7 --num-envs 256 --episodes 4 \
      --checkpoint logs/amr_lab/caster4/<run>/model_2999.pt --out /tmp/by_active.csv

``play.py`` prints the campaign's table and nothing else, which is enough to rank
conditions and not enough to say why one of them failed. Two things are missing from it.

The first is the scenario. Episodes are sampled from the same generator the policy is
stepping, so two conditions never see the same ones: a jackknife ends an episode early,
the next reset falls on a different step, and the streams have diverged by the second
episode. A difference of a few points between two conditions then has an unknown
component that is neither evaluation noise nor a difference between the policies, and the
pairing that would remove it is not available. ``--scenario-seed`` makes the episode a
lookup keyed by its index, so every condition runs the same scenarios and the comparison
can be made pair by pair --- both succeed, one succeeds, neither does.

The second is the cause of a failure. The table reports the pose error of the episodes
that *reached* the goal, so a condition that fails differently rather than more often
looks the same in it. Here every episode carries the reason it ended --- reached,
jackknife, wall contact, the cart tipped, or the clock ran out --- together with the
geometry it was given, so the failures can be counted by cause and read against the
scenario that produced them.

The rows are written as CSV and analysed separately; this script only prints a summary
of the outcome counts so that a run can be sanity-checked without opening the file.
"""

import argparse
import csv
import os
import sys

from isaaclab.app import AppLauncher

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cli  # noqa: E402, I100

parser = argparse.ArgumentParser(description=__doc__)
cli.add_arguments(parser)
parser.add_argument('--checkpoint', type=str, default=None)
parser.add_argument('--episodes', type=int, default=4,
                    help='episodes per environment; with --scenario-seed this is also '
                         'the number of scenarios, episodes * num_envs of them')
parser.add_argument('--out', type=str, required=True, help='CSV to write')
parser.add_argument('--label', type=str, default='', help='condition name for the CSV')
parser.add_argument('--max-steps', type=int, default=6000,
                    help='give up waiting for the last scenarios after this many steps')
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import math  # noqa: E402, I100

import torch  # noqa: E402, I100
from isaaclab_rl.rsl_rl import handle_deprecated_rsl_rl_cfg, RslRlVecEnvWrapper  # noqa: E402, I100, E501
from rsl_rl.runners import OnPolicyRunner  # noqa: E402, I100

FIELDS = ['label', 'scenario', 'order', 'env', 'cause', 'reached', 'time', 'final_pos',
          'final_yaw', 'beta_max', 'beta_end', 'effort', 'effort_max', 'near_limit',
          'entered', 'clearance', 'back_dist', 'fwd_dist', 'tail_along', 'start_x',
          'start_y', 'start_yaw', 'goal_x', 'goal_y', 'goal_yaw', 'goal_bearing',
          'goal_turn', 'payload', 'friction', 'hitch_limit', 'rolling', 'scrub',
          'entry_t', 'entry_py', 'entry_yaw', 'entry_beta', 'entry_clear',
          'track_err', 'track_max', 'sat_t']


def main():
    import importlib.metadata as metadata

    from amr_lab.agents.rsl_rl_ppo_cfg import TowingPPORunnerCfg
    from amr_lab.towing_env import TowingEnv

    if args.num_envs is None:
        args.num_envs = 256
    env_cfg, _ = cli.build_cfg(args)
    env_cfg.sim.device = args.device
    agent_cfg = TowingPPORunnerCfg()
    agent_cfg.device = args.device
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version('rsl-rl-lib'))

    raw = TowingEnv(env_cfg, render_mode=None)
    env = RslRlVecEnvWrapper(raw, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    if args.checkpoint:
        runner.load(os.path.abspath(args.checkpoint))
        policy = runner.get_inference_policy(device=raw.device)
        print(f'[amr_lab] policy {args.checkpoint}')
    else:
        policy = None
        print('[amr_lab] no checkpoint: holding zero action')

    n, dev = raw.num_envs, raw.device
    target = args.episodes * n
    beta_max = torch.zeros(n, device=dev)
    effort_sum = torch.zeros(n, device=dev)
    effort_max = torch.zeros(n, device=dev)
    near_limit = torch.zeros(n, device=dev)
    steps = torch.zeros(n, device=dev)
    back_dist = torch.zeros(n, device=dev)
    fwd_dist = torch.zeros(n, device=dev)
    # The state of the deck the first time it crossed the mouth plane between the walls.
    # A bay episode fails either before the entry or after it, and the two failures are
    # different behaviours: an entry taken at a large lateral offset or with the cart
    # still swung across the slot axis has little room left to correct, and that is what
    # the conditional contact rate is made of. -1 means the deck never entered, and also
    # covers the episode whose entry falls on its own last step: the reset has already
    # moved the cart by the time this loop can read it, so only the time is honest there.
    entry_t = torch.full((n,), -1.0, device=dev)
    entry_py = torch.zeros(n, device=dev)
    entry_yaw = torch.zeros(n, device=dev)
    entry_beta = torch.zeros(n, device=dev)
    entry_clear = torch.zeros(n, device=dev)
    # How well the servo holds the angle it is commanded, and when it first runs out of
    # torque. For the servo-held condition the command is a constant zero, so this is the
    # angle the finite servo gives away to the cart; for the active one it is the error
    # against a moving target. The passive and the mechanically locked joints have no
    # command, so the column is the deflection from aligned instead.
    track_sum = torch.zeros(n, device=dev)
    track_max = torch.zeros(n, device=dev)
    sat_t = torch.full((n,), -1.0, device=dev)
    TAIL = 20
    tail = torch.zeros(n, TAIL, device=dev)
    tail_at = 0
    # What each environment was handed at the start of the episode it is running. Read
    # after the reset and held, because the goal is resampled before the row is written.
    start = torch.zeros(n, 3, device=dev)
    goal = raw.goal.clone()
    params = raw.params.clone()
    scenario = raw.scenario_id.clone()
    rows = {}
    # Every episode in the order it finished. `play.py` keeps the first `episodes *
    # num_envs` of these, and an environment that terminates early gets through more
    # episodes than one that runs to the clock, so that order is not a sample of the
    # scenario distribution: the 20 s timeouts are the slowest episodes there are and
    # arrive last. Recorded here so the two estimators can be compared on one rollout.
    arrivals = []

    def snapshot(ids):
        pose = raw.robot.data.root_pos_w[ids, :2] - raw.scene.env_origins[ids, :2]
        q = raw.robot.data.root_quat_w[ids]
        yaw = torch.atan2(2.0 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]),
                          1.0 - 2.0 * (q[:, 2] ** 2 + q[:, 3] ** 2))
        start[ids, :2] = pose
        start[ids, 2] = yaw
        goal[ids] = raw.goal[ids]
        params[ids] = raw.params[ids]
        scenario[ids] = raw.scenario_id[ids]

    snapshot(torch.arange(n, device=dev))
    obs = env.get_observations()
    with torch.inference_mode():
        for step in range(args.max_steps):
            action = (policy(obs) if policy is not None else
                      torch.zeros(n, raw.cfg.action_space, device=dev))
            obs, _, dones, _ = env.step(action)
            beta = raw.robot.data.joint_pos[:, raw._yaw]
            beta_max = torch.maximum(beta_max, beta.abs())
            tau = raw.hitch_torque.abs()
            effort_sum += tau
            effort_max = torch.maximum(effort_max, tau)
            # Against the ceiling in force at this speed, which equals the flat
            # limit unless the DC torque-speed curve is in use.
            near_limit += (tau > 0.9 * raw.hitch_ceiling).float()
            steps += 1
            driven = env_cfg.active_hitch or env_cfg.lock_hitch
            err = (raw.beta_target - beta).abs() if driven else beta.abs()
            track_sum += err
            track_max = torch.maximum(track_max, err)
            now = steps * raw.step_dt
            first_sat = (sat_t < 0) & (tau > 0.9 * raw.hitch_limit)
            sat_t = torch.where(first_sat, now, sat_t)
            cart_pos, cart_yaw = raw._cart_pose_true()
            if env_cfg.bay:
                fresh = raw.last_entered & (entry_t < 0) & ~dones.bool()
                entry_t = torch.where(fresh, now, entry_t)
                entry_py = torch.where(fresh, cart_pos[:, 1], entry_py)
                entry_yaw = torch.where(
                    fresh, torch.atan2(torch.sin(cart_yaw - raw.goal[:, 2]),
                                       torch.cos(cart_yaw - raw.goal[:, 2])).abs(),
                    entry_yaw)
                entry_beta = torch.where(fresh, beta.abs(), entry_beta)
                # `wall_gap` is already the minimum over the four deck corners.
                entry_clear = torch.where(fresh, raw.wall_gap, entry_clear)
            cart_v = raw.robot.data.body_lin_vel_w[:, raw._cart, :2]
            along = (cart_v[:, 0] * torch.cos(cart_yaw)
                     + cart_v[:, 1] * torch.sin(cart_yaw)) * raw.step_dt
            back_dist += along.clamp(max=0.0).abs()
            fwd_dist += along.clamp(min=0.0)
            tail[:, tail_at % TAIL] = along
            tail_at += 1
            finished = dones.bool().nonzero(as_tuple=False).flatten()
            for i in finished.tolist():
                # The cause, in the order the termination applies it: contact beats the
                # goal on the step both happen (see _get_dones), and a timeout is only a
                # cause when nothing else ended the episode.
                if bool(raw.last_jackknife[i]):
                    cause = 'jackknife'
                elif bool(raw.last_tipped[i]):
                    cause = 'tipped'
                elif bool(raw.last_contact[i]):
                    cause = 'wall'
                elif bool(raw.last_reached[i]):
                    cause = 'reached'
                elif bool(raw.last_timeout[i]):
                    cause = 'timeout'
                else:
                    cause = 'other'
                sx, sy, syaw = start[i].tolist()
                gx, gy, gyaw = goal[i].tolist()
                bearing = math.atan2(gy - sy, gx - sx)
                key = int(scenario[i].item())
                row = {
                    'label': args.label, 'scenario': key, 'order': len(arrivals),
                    'env': i, 'cause': cause,
                    'reached': int(cause == 'reached'),
                    'time': steps[i].item() * raw.step_dt,
                    'final_pos': raw.last_distance[i].item(),
                    'final_yaw': abs(raw.last_yaw_error[i].item()),
                    'beta_max': beta_max[i].item(),
                    'beta_end': raw.last_beta[i].item(),
                    'effort': (effort_sum[i] / steps[i]).item(),
                    'effort_max': effort_max[i].item(),
                    'near_limit': 100.0 * (near_limit[i] / steps[i]).item(),
                    'entered': int(bool(raw.last_entered[i].item())),
                    'clearance': raw.last_clearance[i].item(),
                    'back_dist': back_dist[i].item(),
                    'fwd_dist': fwd_dist[i].item(),
                    'tail_along': tail[i].sum().item(),
                    'start_x': sx, 'start_y': sy, 'start_yaw': syaw,
                    'goal_x': gx, 'goal_y': gy, 'goal_yaw': gyaw,
                    # How far round the goal pose is from the straight line to it: the
                    # free-space task's difficulty is in this angle, not in the distance.
                    'goal_bearing': bearing,
                    'goal_turn': abs(math.atan2(math.sin(gyaw - bearing),
                                                math.cos(gyaw - bearing))),
                    'payload': params[i, 0].item() * 100.0,
                    'friction': params[i, 8].item(),
                    'hitch_limit': params[i, 4].item(),
                    'rolling': params[i, 5].item(),
                    'scrub': params[i, 6].item(),
                    'entry_t': entry_t[i].item(),
                    'entry_py': entry_py[i].item(),
                    'entry_yaw': entry_yaw[i].item(),
                    'entry_beta': entry_beta[i].item(),
                    'entry_clear': entry_clear[i].item(),
                    'track_err': (track_sum[i] / steps[i]).item(),
                    'track_max': track_max[i].item(),
                    'sat_t': sat_t[i].item(),
                }
                # First occurrence wins, so a scenario is recorded once however many
                # times the environment comes back round to its index.
                rows.setdefault(key, row)
                arrivals.append(row)
            if len(finished):
                for buf in (beta_max, effort_sum, effort_max, near_limit, steps,
                            back_dist, fwd_dist, track_sum, track_max, entry_py,
                            entry_yaw, entry_beta, entry_clear):
                    buf[finished] = 0.0
                entry_t[finished] = -1.0
                sat_t[finished] = -1.0
                tail[finished] = 0.0
                snapshot(finished)
            if env_cfg.scenario_seed is not None:
                have = sum(1 for k in rows if 0 <= k < target)
            else:
                have = len(rows)
            if have >= target:
                break

    done = [rows[k] for k in sorted(rows) if 0 <= k < target]
    with open(args.out, 'w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        for row in done:
            writer.writerow(row)
    first = arrivals[:target]
    if len(first) >= target:
        a = sum(1 for r in first if r['cause'] == 'reached')
        b = sum(1 for r in done if r['cause'] == 'reached')
        print(f'  arrival order would report {a}/{len(first)} '
              f'({100.0 * a / len(first):.1f}%); one row per scenario reports '
              f'{b}/{len(done)} ({100.0 * b / max(1, len(done)):.1f}%)')
    counts = {}
    for row in done:
        counts[row['cause']] = counts.get(row['cause'], 0) + 1
    print(f'\n=== {args.label or "run"}: {len(done)}/{target} scenarios, '
          f'curriculum {env_cfg.curriculum}, scenario seed {env_cfg.scenario_seed}')
    for cause in ('reached', 'timeout', 'wall', 'jackknife', 'tipped', 'other'):
        if counts.get(cause):
            print(f'  {cause:<10} {counts[cause]:5d} '
                  f'({100.0 * counts[cause] / max(1, len(done)):5.1f}%)')
    print(f'  written to {args.out}')
    env.close()


if __name__ == '__main__':
    main()
    simulation_app.close()
    os._exit(0)
