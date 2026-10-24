"""PPO settings for the towing task.

The networks are deliberately small. The observation is 56 numbers of proprioception
and a goal, the action is three, and the thing that has to be learnt is a coupling
between two rigid bodies - not a gait. A wider net mostly buys a longer wall clock.

The actor and the critic see different things (see ``towing_env``), which is what
``obs_groups`` sets up: the actor gets the policy group, the critic gets that plus the
privileged one.
"""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg,
                                RslRlPpoAlgorithmCfg)


@configclass
class TowingPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 32
    max_iterations = 3000
    save_interval = 100
    experiment_name = 'amr_towing'
    empirical_normalization = False
    obs_groups = {'policy': ['policy'], 'critic': ['critic']}
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.8,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation='elu',
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule='adaptive',
        gamma=0.995,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
