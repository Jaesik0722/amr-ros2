"""Roll a trained policy out and print the numbers the campaign reports.

  python amr_lab/play.py --headless --checkpoint logs/amr_lab/trailer/<run>/model_1500.pt
  python amr_lab/play.py --checkpoint ... --num-envs 32          # watch it

The table is deliberately the one in ``~/amr_results/report/results.md``: success rate,
time, final pose error, hitch angle and hitch effort. The passive numbers in E5 - 9/10
finished, 68.7 s, off-tracking rms 220 mm, beta at the 90 deg limit on every run - are
what these are read against, so they have to be the same quantities.

The breakdowns are by payload and by whether the goal needed the cart moved backwards,
because those are the two axes where the active hitch is expected to pay or to fail.

A goal behind the cart can in principle be reached without ever moving the cart
backwards: the robot can swing the cart round and drive to it forwards. Reaching such a
goal is therefore not evidence of reversing, so the cart's travel is split into the part
run along its own heading and the part run against it, and the reversing column reports
the latter. A reversing claim rests on that column, not on the goal split.
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cli  # noqa: E402, I100

parser = argparse.ArgumentParser(description=__doc__)
cli.add_arguments(parser)
parser.add_argument('--checkpoint', type=str, default=None,
                    help='trained model; without one the policy is the untrained mean '
                         '(zero action), which is a useful sanity check of the task')
parser.add_argument('--episodes', type=int, default=8, help='episodes per environment')
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import math  # noqa: E402, I100

import torch  # noqa: E402, I100
from isaaclab_rl.rsl_rl import handle_deprecated_rsl_rl_cfg, RslRlVecEnvWrapper  # noqa: E402, I100
from rsl_rl.runners import OnPolicyRunner  # noqa: E402, I100


def summarise(label, rows):
    """One line of the table from a list of per-episode dicts."""
    if not rows:
        return f'  {label:<22} -'
    n = len(rows)
    ok = [r for r in rows if r['reached']]

    def stat(key, source=rows, scale=1.0):
        values = torch.tensor([r[key] for r in source], dtype=torch.float)
        if not len(values):
            return 'n/a'
        # A single sample has no sample standard deviation, and torch warns on stderr if
        # asked for one. The campaign merges stderr into the log it parses, so that
        # warning would land inside the table and shift every column after it.
        mean = values.mean().item() * scale
        sd = values.std().item() * scale if len(values) > 1 else 0.0
        return f'{mean:7.1f} +- {sd:5.1f}'

    return (f'  {label:<22} {len(ok):4d}/{n:<4d} '
            f'{stat("time", ok):>18} {stat("final_pos", ok, 1000):>18} '
            f'{stat("final_yaw", ok, 180 / math.pi):>18} '
            f'{stat("beta_max", rows, 180 / math.pi):>18} '
            f'{stat("effort", rows):>18} '
            f'{stat("effort_max", rows):>18} '
            f'{stat("back_dist", ok):>18} '
            f'{sum(r["jackknife"] for r in rows):4d}')


def main():
    import importlib.metadata as metadata

    from amr_lab.agents.rsl_rl_ppo_cfg import TowingPPORunnerCfg
    from amr_lab.towing_env import TowingEnv

    if args.num_envs is None:
        args.num_envs = 256
    env_cfg, asset_path = cli.build_cfg(args)
    env_cfg.sim.device = args.device
    agent_cfg = TowingPPORunnerCfg()
    agent_cfg.device = args.device
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version('rsl-rl-lib'))

    raw = TowingEnv(env_cfg, render_mode=None)
    env = RslRlVecEnvWrapper(raw, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    if args.checkpoint:
        runner.load(os.path.abspath(args.checkpoint))
        policy = runner.get_inference_policy(device=env.unwrapped.device)
        print(f'[amr_lab] policy {args.checkpoint}')
    else:
        policy = None
        print('[amr_lab] no checkpoint: holding zero action')

    n, dev = raw.num_envs, raw.device
    # Per-episode accumulators. An episode ends when the env resets, so the stats are
    # closed out on the step the done flag comes back.
    beta_max = torch.zeros(n, device=dev)
    effort_sum = torch.zeros(n, device=dev)
    steps = torch.zeros(n, device=dev)
    # Path length the cart ran against its own heading, and with it. Accumulated from the
    # cart's world velocity projected on its heading, so a robot that turns the cart
    # round and drives forwards registers no reverse travel however the goal was placed.
    back_dist = torch.zeros(n, device=dev)
    fwd_dist = torch.zeros(n, device=dev)
    # Signed along-heading travel over the last second of the episode, in a ring. Total
    # reverse travel does not say a cart was *backed into place*: an episode can reverse
    # half a metre early on and then swing round and drive in forwards. A reverse-docking
    # claim is about the final approach, so the last second of it is kept separately.
    TAIL = 20                               # control steps; 1.0 s at 20 Hz
    tail = torch.zeros(n, TAIL, device=dev)
    tail_at = 0
    # Peak as well as mean: a servo is specified by both, and a mean well inside the
    # stall rating says nothing about whether the peak sits on it.
    effort_max = torch.zeros(n, device=dev)
    # How fast the hitch was turning at the moment it asked for the most torque, and how
    # much of the episode it spent near the limit. A flat effort limit is generous to a
    # DC servo exactly when the joint is moving, so a peak demand taken at speed is a
    # demand the real servo cannot meet; a peak taken near standstill is one it can.
    rate_at_peak = torch.zeros(n, device=dev)
    near_limit = torch.zeros(n, device=dev)
    # Whether this goal needs the cart moved backwards: the goal is behind the cart's
    # start, which is where the robot has to push rather than pull.
    backward = raw.goal[:, 0] < -0.85
    payload = raw.params[:, 0].clone() * 100.0
    rows = []

    obs, _ = env.get_observations(), None
    target = args.episodes * n
    with torch.inference_mode():
        while len(rows) < target:
            action = (policy(obs) if policy is not None else
                      torch.zeros(n, raw.cfg.action_space, device=dev))
            obs, _, dones, _ = env.step(action)
            beta = raw.robot.data.joint_pos[:, raw._yaw]
            beta_max = torch.maximum(beta_max, beta.abs())
            effort_sum += raw.hitch_torque.abs()
            tau = raw.hitch_torque.abs()
            rate = raw.robot.data.joint_vel[:, raw._yaw].abs()
            peak = tau > effort_max
            rate_at_peak = torch.where(peak, rate, rate_at_peak)
            effort_max = torch.maximum(effort_max, tau)
            near_limit += (tau > 0.9 * raw.hitch_limit).float()
            steps += 1
            _, cart_yaw = raw._cart_pose_true()
            cart_v = raw.robot.data.body_lin_vel_w[:, raw._cart, :2]
            along = (cart_v[:, 0] * torch.cos(cart_yaw)
                     + cart_v[:, 1] * torch.sin(cart_yaw)) * raw.step_dt
            back_dist += along.clamp(max=0.0).abs()
            fwd_dist += along.clamp(min=0.0)
            tail[:, tail_at % TAIL] = along
            tail_at += 1
            # The env records the terminal error before it resamples the goal, so read
            # it from there rather than recomputing it against the new goal.
            finished = dones.bool().nonzero(as_tuple=False).flatten()
            for i in finished.tolist():
                rows.append({
                    'reached': bool(raw.last_reached[i]),
                    'jackknife': int(raw.last_jackknife[i]),
                    'time': steps[i].item() * raw.step_dt,
                    'final_pos': raw.last_distance[i].item(),
                    'final_yaw': abs(raw.last_yaw_error[i].item()),
                    'beta_max': beta_max[i].item(),
                    'effort': (effort_sum[i] / steps[i]).item(),
                    'backward': bool(backward[i]),
                    'effort_max': effort_max[i].item(),
                    'rate_at_peak': rate_at_peak[i].item(),
                    'near_limit': 100.0 * (near_limit[i] / steps[i]).item(),
                    'back_dist': back_dist[i].item(),
                    'fwd_dist': fwd_dist[i].item(),
                    'tail_along': tail[i].sum().item(),
                    'clearance': raw.last_clearance[i].item(),
                    'entered': bool(raw.last_entered[i].item()),
                    'payload': payload[i].item(),
                })
            if len(finished):
                beta_max[finished] = 0.0
                effort_sum[finished] = 0.0
                steps[finished] = 0.0
                back_dist[finished] = 0.0
                fwd_dist[finished] = 0.0
                effort_max[finished] = 0.0
                rate_at_peak[finished] = 0.0
                near_limit[finished] = 0.0
                tail[finished] = 0.0
                backward[finished] = raw.goal[finished, 0] < -0.85
                payload[finished] = raw.params[finished, 0] * 100.0

    rows = rows[:target]
    print(f'\n=== {env_cfg.cart_wheels} cart, '
          f'{"active" if env_cfg.active_hitch else "passive"} hitch, '
          f'{len(rows)} episodes, payload {env_cfg.payload_mass_range} kg')
    print(f'  {"":<22} {"reached":>9} {"time [s]":>18} {"pos err [mm]":>18} '
          f'{"yaw err [deg]":>18} {"beta max [deg]":>18} {"hitch [Nm]":>18} '
          f'{"hitch peak [Nm]":>18} {"cart back [m]":>18} {"jack":>4}')
    print(summarise('all', rows))
    print(summarise('goal ahead', [r for r in rows if not r['backward']]))
    print(summarise('goal behind the cart', [r for r in rows if r['backward']]))
    import statistics as _st
    rp = [r['rate_at_peak'] for r in rows]
    nl = [r['near_limit'] for r in rows]
    tau_avail = [raw.cfg.hitch_torque * max(0.0, 1.0 - r / raw.cfg.hitch_noload_speed)
                 for r in rp]
    print(f'  hitch at its peak demand: rate {_st.mean(rp):.2f} rad/s, so a DC servo of '
          f'{raw.cfg.hitch_torque:.1f} N.m stall and {raw.cfg.hitch_noload_speed:.2f} '
          f'rad/s no-load speed could supply {_st.mean(tau_avail):.1f} N.m there; '
          f'{_st.mean(nl):.1f}% of each episode is spent above 90% of the limit')
    behind_ok = [r for r in rows if r['backward'] and r['reached']]
    if behind_ok:
        # A goal behind the cart reached without reversing was reached by turning the
        # cart round, which is a different manoeuvre and not what this paper claims.
        reversed_ = [r for r in behind_ok if r['back_dist'] > 0.5]
        swung = [r for r in behind_ok if r['fwd_dist'] > 2 * max(r['back_dist'], 1e-6)]
        # The final second, separately: this is the one that supports "backed into
        # place" rather than "reversed at some point on the way".
        tail_rev = [r for r in behind_ok if r['tail_along'] < -0.05]
        print(f'  of the {len(behind_ok)} reached goals behind the cart: '
              f'{len(reversed_)} ({100.0 * len(reversed_) / len(behind_ok):.1f}%) '
              f'backed it more than 0.5 m, '
              f'{len(swung)} ({100.0 * len(swung) / len(behind_ok):.1f}%) '
              f'ran mostly forwards, '
              f'{len(tail_rev)} ({100.0 * len(tail_rev) / len(behind_ok):.1f}%) '
              f'were still reversing over the last second')
    if env_cfg.bay:
        # Geometric clearance, not a contact sensor: see _track_clearance. 10 m is the
        # sentinel for a cart that never entered the slot at all, so those episodes are
        # counted separately rather than averaged in as "plenty of room".
        entered = [r for r in rows if r['entered']]
        touched = [r for r in entered
                   if r['clearance'] < raw.cfg.bay_contact_clearance]
        gaps = torch.tensor([r['clearance'] for r in entered], dtype=torch.float)
        print(f'  bay {raw.cfg.bay_width * 1000:.0f} mm clear span, deck 560 mm: '
              f'{len(entered)}/{len(rows)} episodes put the deck inside the slot, '
              f'min clearance '
              f'{(gaps.mean().item() * 1000 if len(gaps) else float("nan")):.0f} +- '
              f'{(gaps.std().item() * 1000 if len(gaps) > 1 else 0.0):.0f} mm, '
              f'{len(touched)} touched a wall')
        # Contact ends the episode and is a failure, so wall contact is a failure mode
        # alongside the jackknife rather than something a success can contain. The
        # agreement is asserted rather than assumed: `reached` and `touched` are
        # computed from different quantities, and if they ever overlap the termination
        # and the report have drifted apart, which is the bug this task had.
        ok = [r for r in rows if r['reached']]
        both = [r for r in ok if r['clearance'] < raw.cfg.bay_contact_clearance]
        print(f'  wall contact ended {len(touched)}/{len(rows)} episodes '
              f'({100.0 * len(touched) / max(1, len(rows)):.1f}%), against '
              f'{sum(r["jackknife"] for r in rows)} jackknifed; '
              f'{len(ok)} placed the cart with the walls clear'
              + (f'  [!! {len(both)} counted as both]' if both else ''))
    lo, hi = env_cfg.payload_mass_range
    edges = [lo + (hi - lo) * k / 4 for k in range(5)]
    for k in range(4):
        band = [r for r in rows if edges[k] <= r['payload'] < edges[k + 1] + 1e-6]
        print(summarise(f'payload {edges[k]:.0f}-{edges[k + 1]:.0f} kg', band))
    env.close()


if __name__ == '__main__':
    main()
    simulation_app.close()
