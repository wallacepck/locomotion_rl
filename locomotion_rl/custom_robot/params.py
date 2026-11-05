"""PPO hyper-parameters for the custom robot (counterpart of mujoco_playground.config.locomotion_params.brax_ppo_config).

Differences from the Go1 recipe (200M steps, 8192 envs, 512-256-128 networks), see the comments on each line:
only the policy network is smaller, because it has to run on the robot's controller eventually. Env count and step budget
are NOT derated: measured on the RTX 4060 laptop GPU (8 GB), memory is ~2.3 GB at 2048-8192 envs (the PPO batch is fixed).
"""
from ml_collections import config_dict

# Contacts one environment can produce: 4 feet x up to 2 cylinder-edge contacts each, plus margin.
# warp sizes its contact buffer for the whole batch, so naconmax = NACON_PER_ENV * number of worlds.
NACON_PER_ENV = 10
# GJK/EPA slots, used only where a foot meets a heightfield (the flat plane is analytic). Warp allocates ~10 scratch
# buffers of this many rows *inside every step*, so the default (= naconmax) means a ~35 MB allocation per step at 8192 envs.
CCD_PER_ENV = 6


def brax_ppo_config(env_name: str) -> config_dict.ConfigDict:
    from locomotion_rl import registry
    env_cfg = registry.get_default_config(env_name)
    return config_dict.create(
        num_timesteps=200_000_000,         # same as Go1: ~25 min on the laptop GPU (measured ~150k steps/s)
        num_evals=10,
        reward_scaling=1.0,
        episode_length=env_cfg.episode_length,
        normalize_observations=True,
        action_repeat=1,
        unroll_length=20,
        num_minibatches=32,
        num_updates_per_batch=4,
        discounting=0.97,
        learning_rate=3e-4,
        entropy_cost=1e-2,
        num_envs=8192,                     # same as Go1. Measured: peak GPU memory is ~2.3 GB from 2048 to 8192 envs, so no derating.
        batch_size=256,
        max_grad_norm=1.0,
        num_eval_envs=128,
        num_resets_per_eval=10,
        network_factory=config_dict.create(
            policy_hidden_layer_sizes=(256, 128, 64),      # Go1: (512, 256, 128). Small enough for the target MCU.
            value_hidden_layer_sizes=(512, 256, 128),      # the critic never leaves the host, keep it Go1-sized
            policy_obs_key="state",
            value_obs_key="privileged_state",
        ),
    )
