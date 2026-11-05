"""Train the custom robot's Joystick policy with Brax PPO (script version of Playground's locomotion.ipynb).

    PYTHONPATH=src/locomotion_rl venv/bin/python -m locomotion_rl.locomotion --num_timesteps 2_000_000 --num_envs 1024

Anything not given on the command line comes from custom_robot/params.py.
"""
import argparse
import ast
import csv
import json
import os
from datetime import datetime

# Must be set before jax is imported.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")   # warp shares the GPU with jax; do not let jax take it all
os.environ.setdefault("MUJOCO_GL", "egl")

import functools
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
from orbax import checkpoint as ocp
from flax.training import orbax_utils
from mujoco_playground import wrapper

from locomotion_rl import compat  # noqa: F401  (jax/brax version shim, See: https://github.com/google/brax/issues/671)

from locomotion_rl import registry

np.set_printoptions(precision=3, suppress=True, linewidth=100)

parser = argparse.ArgumentParser()
parser.add_argument("--env_name", default="CustomRobotJoystickFlatTerrain")
parser.add_argument("--num_timesteps", type=int)
parser.add_argument("--num_envs", type=int)
parser.add_argument("--num_evals", type=int)
parser.add_argument("--impl", default="warp", choices=["warp", "jax"])
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--no_randomization", action="store_true")
parser.add_argument("--logdir", default="logs")
parser.add_argument("--env_override", nargs="*", default=[], metavar="KEY=VALUE",
                    help="env config overrides, e.g. reward_config.tracking_sigma=0.03 (floats/ints/strings)")
parser.add_argument("--plot_rewards", nargs="*", default=None, metavar="TERM",
                    help="also write rewards.png with these reward terms (all terms with no names); metrics.csv is always written")
parser.add_argument("--energy_schedule", nargs=2, type=float, metavar=("START", "END"),
                    help="ramp the energy weight from START x to END x (geometric) over the whole run, e.g. 1 10")
args = parser.parse_args()

env_name = args.env_name


def _lookup(cfg, dotted):
    for part in dotted.split("."):
        cfg = cfg[part]
    return cfg


env_cfg = registry.get_default_config(env_name)
env_cfg.impl = args.impl
for item in args.env_override:
    key, value = item.split("=", 1)
    old = _lookup(env_cfg, key)
    new = type(old)(value) if old is not None and not isinstance(old, bool) else ast.literal_eval(value)
    env_cfg.update_from_flattened_dict({key: new})

ppo_params = registry.get_ppo_config(env_name)
for key in ("num_timesteps", "num_envs", "num_evals"):
    if getattr(args, key) is not None:
        ppo_params[key] = getattr(args, key)
print(f"PPO parameters:\n{ppo_params}")

sched = env_cfg.reward_config.energy_schedule
if args.energy_schedule:
    sched.start, sched.end = args.energy_schedule
    sched.ramp_steps = ppo_params.num_timesteps // ppo_params.num_envs         # per-env steps
    print(f"energy weight x{sched.start:g} -> x{sched.end:g} over {ppo_params.num_timesteps:,} steps ({sched.ramp_steps:,} per env)")


def energy_scale_at(num_steps):
    return sched.start * (sched.end / sched.start) ** min(num_steps / ppo_params.num_timesteps, 1.0)


eval_env = registry.load(env_name, config=env_cfg, num_envs=ppo_params.num_eval_envs)   # evaluates at the final weight
sched.initial_step = 0                                                                   # only the training env ramps
env = registry.load(env_name, config=env_cfg, num_envs=ppo_params.num_envs)
sched.initial_step = None

logdir = Path(args.logdir) / f"{env_name}-{datetime.now():%Y%m%d-%H%M%S}"
ckpt_dir = (logdir / "checkpoints").resolve()
ckpt_dir.mkdir(parents=True, exist_ok=True)
# what rollout.py needs to rebuild the same network and environment from a checkpoint
(logdir / "config.json").write_text(json.dumps({"ppo": ppo_params.to_dict(), "env": env_cfg.to_dict()}, default=list, indent=1))

x_data, y_data, y_dataerr = [], [], []
metrics_file = (logdir / "metrics.csv").open("w", newline="")     # one row per eval; see plot_rewards.py for what the columns mean
metrics_csv = None
times = [datetime.now()]


def progress(num_steps, metrics):
    times.append(datetime.now())
    global metrics_csv
    # Per-term reward contribution over an eval episode (mean over eval episodes), already times ctrl_dt, so the terms add up
    # to the unclipped reward; eval/episode_reward is clipped per step, so it is the larger of the two.
    row = {"steps": num_steps, "episode_reward": float(metrics["eval/episode_reward"]),
           "episode_reward_std": float(metrics["eval/episode_reward_std"]),
           "avg_episode_length": float(metrics["eval/avg_episode_length"]),
           "energy_scale": energy_scale_at(num_steps)}
    for key, value in metrics.items():
        if key.startswith("eval/episode_reward/") and not key.endswith("_std"):
            row["term/" + key.split("/", 2)[2]] = float(value) * env_cfg.ctrl_dt
    if metrics_csv is None:
        metrics_csv = csv.DictWriter(metrics_file, fieldnames=list(row))
        metrics_csv.writeheader()
    metrics_csv.writerow(row)
    metrics_file.flush()
    x_data.append(num_steps)
    y_data.append(metrics["eval/episode_reward"])
    y_dataerr.append(metrics["eval/episode_reward_std"])
    ramp = f"  energy x{energy_scale_at(num_steps):.2f}" if sched.ramp_steps else ""
    print(f"{num_steps:>12,d} steps  reward {y_data[-1]:8.3f} +- {y_dataerr[-1]:.3f}  "
          f"len {metrics['eval/avg_episode_length']:.0f}{ramp}")


def policy_params_fn(current_step, make_policy, params):
    # called at every eval: keep a checkpoint per eval so a bad late run can be rolled back
    ocp.PyTreeCheckpointer().save(
        ckpt_dir / f"{current_step:012d}", params, force=True,
        save_args=orbax_utils.save_args_from_target(params))


randomizer = None if args.no_randomization else registry.get_domain_randomizer(env_name)

ppo_training_params = dict(ppo_params)
network_factory = ppo_networks.make_ppo_networks
if "network_factory" in ppo_params:
    del ppo_training_params["network_factory"]
    network_factory = functools.partial(ppo_networks.make_ppo_networks, **ppo_params.network_factory)

train_fn = functools.partial(
    ppo.train, **ppo_training_params,
    network_factory=network_factory,
    randomization_fn=randomizer,
    progress_fn=progress,
    policy_params_fn=policy_params_fn,
    seed=args.seed,
)

make_inference_fn, params, metrics = train_fn(
    environment=env,
    eval_env=eval_env,
    wrap_env_fn=wrapper.wrap_for_brax_training,
)
print(f"time to jit: {times[1] - times[0]}")
print(f"time to train: {times[-1] - times[1]}")

plt.errorbar(x_data, y_data, yerr=y_dataerr, color="blue")
plt.xlabel("# environment steps")
plt.ylabel("reward per episode")
plt.savefig(logdir / "reward.png", dpi=120)
metrics_file.close()
if args.plot_rewards is not None:
    from locomotion_rl import plot_rewards
    plot_rewards.plot([logdir], args.plot_rewards or None, logdir / "rewards.png")
print(f"checkpoints and reward.png in {logdir}")
