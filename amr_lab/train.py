"""Train the towing policy.

  python amr_lab/train.py --headless --num-envs 2048 --max-iterations 1500
  python amr_lab/train.py --headless --cart-wheels caster4 --payload 0 60
  python amr_lab/train.py --headless --passive-hitch --run-name baseline

The passive-hitch run is the baseline: the same task, the same reward, the same
randomisation, with the hitch servo limp. That is the system the campaign in
``amr_experiments`` measured, so the comparison is against a number that already exists
rather than against an untuned controller written for the purpose.
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cli  # noqa: E402, I100

parser = argparse.ArgumentParser(description=__doc__)
cli.add_arguments(parser)
parser.add_argument('--resume', type=str, default=None, help='checkpoint to continue from')
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# ---- everything below needs the app ---------------------------------------------------
import time  # noqa: E402, I100

import torch  # noqa: E402, I100
from isaaclab.utils.dict import print_dict  # noqa: E402, I100
from isaaclab.utils.io import dump_yaml  # noqa: E402, I100
from isaaclab_rl.rsl_rl import handle_deprecated_rsl_rl_cfg, RslRlVecEnvWrapper  # noqa: E402, I100
from rsl_rl.runners import OnPolicyRunner  # noqa: E402, I100

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def main():
    import importlib.metadata as metadata

    from amr_lab.agents.rsl_rl_ppo_cfg import TowingPPORunnerCfg
    from amr_lab.towing_env import TowingEnv

    env_cfg, asset_path = cli.build_cfg(args)
    agent_cfg = TowingPPORunnerCfg()
    agent_cfg.run_name = args.run_name
    if args.max_iterations is not None:
        agent_cfg.max_iterations = args.max_iterations
    if args.seed is not None:
        agent_cfg.seed = args.seed
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args.device
    agent_cfg.device = args.device
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version('rsl-rl-lib'))

    tag = args.cart_wheels + ('' if env_cfg.active_hitch else '_passive')
    log_root = os.path.abspath(os.path.join('logs', 'amr_lab', tag))
    log_dir = os.path.join(log_root, time.strftime('%Y-%m-%d_%H-%M-%S')
                           + (f'_{args.run_name}' if args.run_name else ''))
    print(f'[amr_lab] asset   {asset_path}')
    print(f'[amr_lab] logging {log_dir}')
    print_dict({'cart': env_cfg.cart_wheels, 'active hitch': env_cfg.active_hitch,
                'payload [kg]': env_cfg.payload_mass_range,
                'hitch limit [Nm]': env_cfg.hitch_effort_range,
                'envs': env_cfg.scene.num_envs,
                'obs': env_cfg.observation_space, 'states': env_cfg.state_space}, nesting=2)

    env = RslRlVecEnvWrapper(TowingEnv(env_cfg), clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    if args.resume:
        print(f'[amr_lab] resuming from {args.resume}')
        runner.load(args.resume)
    dump_yaml(os.path.join(log_dir, 'params', 'env.yaml'), env_cfg)
    dump_yaml(os.path.join(log_dir, 'params', 'agent.yaml'), agent_cfg)

    start = time.time()
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
    print(f'[amr_lab] training time {time.time() - start:.0f} s')
    env.close()


if __name__ == '__main__':
    main()
    simulation_app.close()
