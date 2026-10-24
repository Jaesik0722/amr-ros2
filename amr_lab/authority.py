"""Find which actuator runs out first as the cart is loaded.

The wheel motors or the hitch servo.

  python amr_lab/authority.py --headless
  python amr_lab/authority.py --headless --payloads 0 20 40 60 80 \
      --hitch-torques 10.6 20 30 --cart-wheels caster4

No policy is involved. Each environment gets one (payload, hitch servo limit) pair and
is driven open loop through three phases, so the answer is a property of the machine
rather than of anything that was learnt:

  1. traction      full speed ahead, hitch straight. Does the robot reach 0.5 m/s, and
                   how much of the MDH300's rated 11 Nm does it take?
  2. steer rolling at 0.3 m/s, command the hitch to 1.0 rad. How far does it get?
  3. steer at rest standing still, command the hitch to 1.0 rad. How far does it get?

Phase 3 is the one that matters for the payload question. Turning the cart on the spot
means overcoming the casters sideways - roughly mu*m*g on a lever of about 0.47 m - which
is tens of Nm even for an empty cart, far past any servo that fits on this hitch. If
phase 3 goes nowhere while phase 2 works, then the active hitch steers only a cart that
is already rolling, and that is a statement about the mechanism, not about tuning.
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cli  # noqa: E402, I100

parser = argparse.ArgumentParser(description=__doc__)
cli.add_arguments(parser)
parser.add_argument('--payloads', type=float, nargs='+',
                    default=[0.0, 20.0, 40.0, 60.0, 80.0, 100.0],
                    help='payload masses to test [kg], on top of the 22.6 kg cart')
parser.add_argument('--hitch-torques', type=float, nargs='+', default=[10.6],
                    help='hitch servo effort limits to test [Nm]')
parser.add_argument('--scrubs', type=float, nargs='+', default=[0.7],
                    help='caster swivel scrub coefficients to test (caster4 only); this '
                         'is the dominant unknown for the real cart')
parser.add_argument('--frictions', type=float, nargs='+', default=[0.8],
                    help='tyre-floor static friction coefficients to test')
parser.add_argument('--wheel-torques', type=float, nargs='+', default=[11.0],
                    help='wheel motor effort limits [Nm]; 11.0 is the MDH300 rating. '
                         'Raising it does not measure the torque the job demands: the '
                         'velocity loop then asks for the whole limit as soon as there '
                         'is any error and breaks traction, so the run turns into noise')
parser.add_argument('--repeats', type=int, default=4)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch  # noqa: E402, I100

TRACTION_S, STEER_S = 4.0, 5.0


def main():
    from amr_lab.towing_env import TowingEnv, WHEEL_RADIUS

    grid = [(m, t, c, f, w) for w in args.wheel_torques for f in args.frictions
            for c in args.scrubs for t in args.hitch_torques for m in args.payloads]
    args.num_envs = len(grid) * args.repeats
    args.payload = [0.0, 0.0]          # the sweep sets the mass itself
    args.hitch_torque = [min(args.hitch_torques), max(args.hitch_torques)]
    env_cfg, _ = cli.build_cfg(args)
    env_cfg.wheel_torque = max(env_cfg.wheel_torque, max(args.wheel_torques))
    env_cfg.robot.actuators['wheels'].effort_limit = env_cfg.wheel_torque
    env_cfg.robot.actuators['wheels'].effort_limit_sim = env_cfg.wheel_torque
    # One payload position for the whole sweep, so mass is the only thing that moves.
    env_cfg.payload_x_range = (0.0, 0.0)
    env_cfg.payload_y_range = (0.0, 0.0)
    env_cfg.payload_z_range = (0.45, 0.45)
    env_cfg.beta_init_range = (0.0, 0.0)
    env_cfg.episode_length_s = 3 * (TRACTION_S + STEER_S)
    # Open loop characterisation, not episodes: nothing here may reset, because a reset
    # resamples the cell's payload and limits and silently averages two machines
    # together. The hitch is commanded hard against its stop, so without this the
    # jackknife termination fires and the readings drift above the servo's own limit.
    env_cfg.jackknife = 100.0
    env = TowingEnv(env_cfg)
    n, dev = env.num_envs, env.device

    def column(k):
        return torch.tensor([g[k] for g in grid],
                            device=dev).repeat_interleave(args.repeats)

    mass, limit, scrub = column(0), column(1), column(2)
    friction, wheel_limit = column(3), column(4)
    env.reset()
    ids = env.robot._ALL_INDICES
    zeros = torch.zeros(n, device=dev)
    env.apply_cart(ids, mass, zeros, zeros, zeros + 0.45,
                   zeros + 0.02, scrub, limit, wheel_limit, friction)
    # Put the goal out of reach so nothing here ends on "reached"; this is open loop.
    env.goal[:, 0] = 1000.0

    def run(seconds, v, beta, ramp=1.0):
        """Drive every env with the same command and return the last-quarter averages.

        The speed command is ramped over `ramp` seconds. The wheels are driven by a
        stiff velocity loop, as the real motor driver is, so a step command asks for the
        whole torque limit on the first control step: with the limit raised to look for
        the torque a heavy cart *demands*, that breaks traction immediately and the
        wheels spin for the rest of the run, which measures nothing.
        """
        action = torch.zeros(n, env_cfg.action_space, device=dev)
        if env_cfg.action_space == 3:
            action[:, 2] = beta
        steps = int(seconds / env.step_dt)
        tail = max(1, steps // 4)
        ramp_steps = max(1, int(ramp / env.step_dt))
        speed = torch.zeros(n, device=dev)
        wheel = torch.zeros(n, device=dev)
        slip = torch.zeros(n, device=dev)
        hitch = torch.zeros(n, device=dev)
        pitch = torch.zeros(n, device=dev)
        beta_max = torch.zeros(n, device=dev)
        for step in range(steps):
            action[:, 0] = (v / env_cfg.action_scale[0]
                            * min(1.0, (step + 1) / ramp_steps))
            env.step(action)
            b = env.robot.data.joint_pos[:, env._yaw]
            beta_max = torch.maximum(beta_max, b.abs())
            if step >= steps - tail:
                body = env.robot.data.root_lin_vel_b[:, 0]
                rim = env.robot.data.joint_vel[:, env._wheels].mean(dim=1) * WHEEL_RADIUS
                speed += body
                # Rim speed minus body speed: how much of the wheel's rotation is going
                # into the floor rather than into moving. This is what separates "the
                # motor ran out" from "the tyres let go".
                slip += rim - body
                wheel += env.robot.data.applied_torque[:, env._wheels].abs().mean(dim=1)
                hitch += env.hitch_torque.abs()
                pitch += env.pitch_torque.abs()
        return (speed / tail, wheel / tail, slip / tail, hitch / tail, beta_max,
                pitch / tail)

    # Phase 1: traction.
    speed, wheel, slip, _, _, pitch = run(TRACTION_S, 0.5, 0.0, ramp=2.0)
    # Phase 2: steer while rolling. The hitch command is reset first so the ramp starts
    # from straight.
    env.beta_target[:] = 0.0
    _, _, _, hitch_roll, beta_roll, _ = run(STEER_S, 0.3, 1.0)
    # Phase 3: steer standing still. Bring the hitch back to straight and let the train
    # stop first, by driving it rather than by teleporting joints - setting joint
    # positions mid-run would drop the cart back behind the robot wherever it actually
    # is, which can put it through the floor.
    run(2.0, 0.0, -1.0)
    env.beta_target[:] = 0.0
    _, _, _, hitch_rest, beta_rest, _ = run(STEER_S, 0.0, 1.0)

    deg = 180.0 / torch.pi
    asked = env_cfg.beta_command_limit
    print(f'\n=== {env_cfg.cart_wheels} cart, hitch commanded to its limit of '
          f'{asked:.2f} rad ({asked * float(deg):.0f} deg), '
          f'{args.repeats} runs per cell')
    print(f'  pitch servo held at {env_cfg.pitch_target:.2f} rad')
    print('  payload  cart hitch_lim scrub   mu |        speed [m/s] |'
          '    rolling: beta [deg]  tau [Nm] |  at rest: beta [deg]')
    print('     [kg]  [kg]      [Nm]    -    -  |      mean +- sd    |'
          '           mean +- sd       mean |        mean +- sd')

    def cell(values, column, scale=1.0):
        v = values[column] * scale
        return f'{v.mean():7.1f} +-{v.std():5.1f}'

    for i, (m, t, c, f, w) in enumerate(grid):
        s = slice(i * args.repeats, (i + 1) * args.repeats)
        print(f'  {m:7.0f} {22.6 + m:5.1f} {t:9.1f} {c:5.2f} {f:4.2f} | '
              f'{speed[s].mean():6.3f} +-{speed[s].std():5.3f} | '
              f'{cell(beta_roll, s, deg)} {hitch_roll[s].mean():10.1f} | '
              f'{cell(beta_rest, s, deg)}')
    print(f'\n  Speed target 0.5 m/s ({0.5 / WHEEL_RADIUS:.2f} rad/s at the wheel), '
          f'ramped over 2 s. slip is rim speed\n  minus body speed: large slip means '
          f'the tyres let go. The wheel torque is not reported\n  because a '
          f'damping-{env_cfg.wheel_damping:.0f} velocity loop saturates on a '
          f'rounding error and says nothing\n  about demand. pitch_tau is the hook '
          f'servo; at its {env_cfg.hitch_torque:.1f} Nm limit it presses the robot'
          f'\n  tail down with about {env_cfg.hitch_torque / 0.0537:.0f} N.')
    env.close()


if __name__ == '__main__':
    main()
    simulation_app.close()
