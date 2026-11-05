"""Roll out a trained policy on the MJX environment and report how well it does, per command.

    PYTHONPATH=src/locomotion_rl venv/bin/python -m locomotion_rl.rollout [--checkpoint logs/<run>] [--gif]

Every scenario (a fixed joystick command) is run on several parallel robots for the same number of control steps, with
the deterministic policy (the mean action, no exploration noise). Observation noise stays on, as in training, unless
--no_noise. Domain randomisation is off unless --randomize, so the default numbers are for the nominal robot.

Per scenario it reports (all over the robots still upright, after a settling period):
    alive      robots that never tipped over (upvector z < 0 is the environment's termination)
    vx vy wz   mean and RMS error of body velocity vs the command
    CoT        sum|tau * qdot| dt / (m * distance): mechanical power in physical units (the 'energy' reward uses the same |tau*qdot| but with tau as a fraction of the stall torque), not electrical
    power      mean sum|tau * qdot| in W (the CoT is meaningless when standing)
    tilt       body tilt from vertical, mean / max
    sat        fraction of joint-steps with |tau| at the torque limit
    >vmax      fraction of joint-steps faster than the servo's rated speed (the model does not enforce it)
    jerk       mean |a_t - a_t-1| of the policy output
    gait       (printed under the table) per scenario, averaged over the feet: cadence = touchdowns per second per foot,
               duty = fraction of time on the floor, clearance = mean height a swinging foot rises above its own liftoff/touchdown level, stride = distance
               the body moves per touchdown. A shuffle shows as high cadence, tiny swing and tiny stride.

Output: everything printed is also written to <run>/<env_name>_rollout[_<tag>].txt, and one row per scenario with all the
numbers to the .csv beside it (see compile_rollouts.py). <env_name> is the env the policy is EVALUATED on (--env_name), so a
checkpoint rolled out on several scenes keeps one set of files per scene; GIFs (--gif) are env-prefixed the same way.
"""
import argparse
import csv
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("MUJOCO_GL", "egl")

import functools

import jax
import jax.numpy as jp
import mujoco
import numpy as np
from brax.training.acme import running_statistics
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.types import UInt64
from ml_collections import config_dict
from orbax import checkpoint as ocp

from locomotion_rl import compat  # noqa: F401  (jax/brax version shim)
from locomotion_rl import metrics, registry
from locomotion_rl.custom_robot.constants import consts

from locomotion_rl.metrics import SCENARIOS  # noqa: E402  (fixed joystick commands, shared with champ_transfer)


def find_checkpoint(path: str | None) -> Path:
    """A run dir, its checkpoints dir, or one step dir -> the step dir (the newest one when several)."""
    root = Path(path) if path else Path("logs")
    if not root.exists():
        raise SystemExit(f"{root} does not exist")
    runs = [root] if (root / "checkpoints").exists() or root.name == "checkpoints" else sorted(p for p in root.iterdir() if p.is_dir())
    for run in reversed(runs):
        ckpts = run if run.name == "checkpoints" else run / "checkpoints"
        steps = sorted(p for p in ckpts.glob("*") if p.is_dir() and p.name.isdigit()) if ckpts.exists() else []
        if steps:
            return steps[-1]
    if root.name.isdigit():
        return root
    raise SystemExit(f"no checkpoint found under {root}")


def merge_saved_config(default, saved: dict):
    """The saved env config laid over the current defaults: values the run used win, keys added to the code since the
    run was saved (e.g. naccdmax) keep their defaults, and keys the code no longer has are dropped with a note."""
    for key, value in saved.items():
        if key not in default:
            print(f"note: saved config key {key!r} no longer exists in the env config; ignored")
        elif isinstance(default[key], config_dict.ConfigDict):
            merge_saved_config(default[key], value)
        else:
            default[key] = value
    return default


def load_policy(step_dir: Path, env, ppo_params):
    """Deterministic policy fn(obs, rng) -> action from a checkpoint saved by locomotion.py."""
    normalizer, policy, value = ocp.PyTreeCheckpointer().restore(step_dir.resolve())
    count = normalizer["count"]
    normalizer = running_statistics.RunningStatisticsState(
        count=UInt64(hi=jp.asarray(count["hi"]), lo=jp.asarray(count["lo"])) if isinstance(count, dict) else count,
        mean=normalizer["mean"], summed_variance=normalizer["summed_variance"], std=normalizer["std"],
        std_eps=normalizer["std_eps"])
    networks = ppo_networks.make_ppo_networks(
        env.observation_size, env.action_size,
        preprocess_observations_fn=running_statistics.normalize, **ppo_params["network_factory"])
    make_policy = ppo_networks.make_inference_fn(networks)
    return make_policy((normalizer, policy, value), deterministic=True)


def render_gif(env, qpos, qvel, path, every=2, size=(480, 360), contact=None, label="", distance=0.45):
    """Tracking-camera GIF of one robot from recorded qpos/qvel (T, nq)/(T, nv).

    Overlay: `label`, the time, and (if `contact` (T, 4) is given) one box per foot in consts.LEGS order,
    green while the foot touches the floor, red while it is in the air.
    """
    from PIL import Image, ImageDraw
    model = env.mj_model
    renderer = mujoco.Renderer(model, height=size[1], width=size[0])
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = model.body(consts.ROOT_BODY).id
    cam.distance, cam.azimuth, cam.elevation = distance, 140.0, -25.0
    data, frames = mujoco.MjData(model), []
    for k in range(0, len(qpos), every):
        data.qpos[:], data.qvel[:] = qpos[k], qvel[k]
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera=cam)
        img = Image.fromarray(renderer.render())
        draw = ImageDraw.Draw(img)
        draw.text((6, 4), f"{label}   t = {k * env.dt:5.2f} s", fill=(0, 0, 0))
        if contact is not None:
            for j, leg in enumerate(consts.LEGS):
                x = 6 + 44 * j
                draw.rectangle([x, 20, x + 40, 34], fill=(60, 180, 75) if contact[k, j] else (220, 60, 50))
                draw.text((x + 6, 21), leg.upper(), fill=(255, 255, 255))
        frames.append(img)
    renderer.close()
    # Compression: one adaptive palette (32 colours) built from frames sampled across the clip, no dithering. A shared palette
    # keeps unchanged pixels identical between frames so PIL can store only the changed region (per-frame palettes did
    # not shrink the file), and dither noise would defeat that too. Measured on a 250-frame clip: 6.9 -> 5.0 MB.
    sample = frames[:: max(len(frames) // 8, 1)][:8]
    mosaic = Image.new("RGB", (sample[0].width, sample[0].height * len(sample)))
    for k, f in enumerate(sample):
        mosaic.paste(f, (0, k * f.height))
    palette = mosaic.quantize(colors=32, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
    frames = [f.quantize(palette=palette, dither=Image.Dither.NONE) for f in frames]
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=int(1000 * env.dt * every), loop=0,
                   optimize=True, disposal=1)


class _Tee:
    """Writes to several streams at once (the console and the results file)."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            stream.write(text)
        return len(text)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", help="run dir, checkpoints dir or step dir (default: newest under ./logs)")
    ap.add_argument("--env_name", default="CustomRobotJoystickFlatTerrain")
    ap.add_argument("--scenarios", nargs="*", default=list(SCENARIOS), help=f"subset of {list(SCENARIOS)}")
    ap.add_argument("--command", nargs=3, type=float, metavar=("VX", "VY", "WZ"), help="one custom scenario instead")
    ap.add_argument("--envs_per_scenario", type=int, default=16)
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--settle", type=float, default=2.0, help="seconds ignored at the start of every velocity statistic")
    ap.add_argument("--impl", default=None, choices=["warp", "jax"])
    ap.add_argument("--randomize", action="store_true", help="apply the training domain randomisation")
    ap.add_argument("--no_noise", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gif", action="store_true", help="write a GIF of the first robot of every scenario to <run>/gifs/")
    ap.add_argument("--gif_scenarios", nargs="*", default=None, metavar="NAME",
                    help="with --gif: only record these scenarios (the table still covers all); default all")
    ap.add_argument("--gif_distance", type=float, default=0.45, help="camera distance in m (the robot is ~0.2 m long)")
    ap.add_argument("--save", help="save the raw per-step records to this .npz")
    ap.add_argument("--tag", default="", help="suffix for the result files, so several rollouts of one run and env can coexist "
                    "(--command runs are tagged custom_<vx>_<vy>_<wz> automatically)")
    args = ap.parse_args()

    step_dir = find_checkpoint(args.checkpoint)
    # Results go to <run>/<env_name>_rollout[_<tag>].txt (what is printed) and .csv (one row per scenario, for
    # compile_rollouts.py). The env name is the EVALUATION env, so one checkpoint can be rolled out on several scenes.
    run_dir = step_dir.parent.parent
    tag = args.tag or ("custom_" + "_".join(f"{v:g}" for v in args.command) if args.command else "")
    stem = f"{args.env_name}_rollout" + (f"_{tag}" if tag else "")
    result_file = (run_dir / f"{stem}.txt").open("w")
    console = sys.stdout
    sys.stdout = _Tee(console, result_file)
    print(f"checkpoint: {step_dir}")
    print(f"evaluation env: {args.env_name}")
    cfg_file = step_dir.parent.parent / "config.json"
    if cfg_file.exists():                     # the config the run was trained with
        saved = json.loads(cfg_file.read_text())
        env_cfg, ppo_params = merge_saved_config(registry.get_default_config(args.env_name), saved["env"]), saved["ppo"]
    else:
        print("no config.json next to the checkpoint: using the registry defaults (network sizes must match)")
        env_cfg, ppo_params = registry.get_default_config(args.env_name), registry.get_ppo_config(args.env_name).to_dict()
    if args.impl:
        env_cfg.impl = args.impl
    if args.no_noise:
        env_cfg.noise_config.level = 0.0

    scenarios = {"custom": tuple(args.command)} if args.command else {k: SCENARIOS[k] for k in args.scenarios}
    names = list(scenarios)
    per, n = args.envs_per_scenario, len(scenarios) * args.envs_per_scenario
    cmds = jp.asarray(np.repeat(np.array([scenarios[k] for k in names]), per, axis=0), dtype=jp.float32)
    steps = int(round(args.seconds / env_cfg.ctrl_dt))

    env = registry.load(args.env_name, config=env_cfg, num_envs=n)
    inference = jax.jit(load_policy(step_dir, env, ppo_params))
    if args.randomize:
        from mujoco_playground._src import wrapper as pg_wrapper
        rnd = registry.get_domain_randomizer(args.env_name)
        venv = pg_wrapper.BraxDomainRandomizationVmapWrapper(
            env, lambda m: rnd(m, jax.random.split(jax.random.PRNGKey(args.seed + 1), n)))
        reset, step = jax.jit(venv.reset), jax.jit(venv.step)
    else:
        reset, step = jax.jit(jax.vmap(env.reset)), jax.jit(jax.vmap(env.step))

    @jax.jit
    def pin_command(state):
        """Force the commanded velocity (the env resamples its own) and rebuild the observation that contains it."""
        info = dict(state.info)
        info["command"] = cmds
        info["steps_until_next_cmd"] = jp.full((n,), 10 ** 9, dtype=jp.int32)
        obs = jax.vmap(lambda d, i: env._get_obs(d, dict(i)))(state.data, info)
        return state.replace(info=info, obs=obs)

    term_keys = sorted(k for k in env_cfg.reward_config.scales.keys())
    term_keys = [f"reward/{k}" for k in term_keys]

    @jax.jit
    def tick(state, rng):
        action, _ = inference(state.obs, rng)
        state = step(state, action)
        d = state.data
        rec = dict(
            action=action, done=state.done,
            linvel=jax.vmap(env.get_local_linvel)(d), yaw_rate=jax.vmap(env.get_global_angvel)(d)[:, 2],
            up_z=jax.vmap(env.get_upvector)(d)[:, 2], tau=d.actuator_force, qd=d.qvel[:, consts.NUM_ROOT_DOF:],
            pos=d.qpos[:, :3], qpos=d.qpos, qvel=d.qvel,
            contact=jp.stack([d.sensordata[:, env.mj_model.sensor_adr[i]] > 0 for i in env._feet_floor_found_sensor], axis=-1),
            foot_z=d.site_xpos[:, env._feet_site_id, 2],
            foot_vxy=jp.linalg.norm(d.sensordata[:, env._foot_linvel_sensor_adr][..., :2], axis=-1),
            terms=jp.stack([state.metrics[k] for k in term_keys], axis=-1), reward=state.reward,
        )
        return state, rec

    state = pin_command(reset(jax.random.split(jax.random.PRNGKey(args.seed), n)))
    rng, recs = jax.random.PRNGKey(args.seed + 2), []
    for _ in range(steps):
        rng, key = jax.random.split(rng)
        state, rec = tick(state, key)
        recs.append(jax.device_get(rec))
    r = {k: np.stack([x[k] for x in recs], axis=1) for k in recs[0]}          # (envs, T, ...)

    if args.save:
        np.savez_compressed(args.save, names=names, per=per, cmds=np.asarray(cmds), **r)

    dt, T = env.dt, steps
    print(f"\n{steps} steps of {dt * 1e3:.0f} ms ({args.seconds:.0f} s), {per} robots per scenario, "
          f"noise {'off' if args.no_noise else 'on'}, randomisation {'on' if args.randomize else 'off'}; "
          f"statistics skip the first {args.settle:g} s and use only robots still upright\n")
    rows, text = metrics.compute(r, names, scenarios, per, dt, args.settle, consts.TOTAL_MASS, consts.TORQUE_LIMIT,
                                 consts.MAX_JOINT_VELOCITY, term_keys)
    print(text)

    meta = dict(env_name=args.env_name, run=run_dir.name, checkpoint_step=step_dir.name, seed=args.seed,
                noise=int(not args.no_noise), randomize=int(args.randomize), seconds=args.seconds, settle=args.settle)
    columns = list(meta) + ["scenario"] + sorted({k for rec in rows.values() for k in rec} - {"scenario"})
    with (run_dir / f"{stem}.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, restval="")
        writer.writeheader()
        for rec in rows.values():
            writer.writerow({**meta, **rec})
    print(f"wrote {run_dir / stem}.txt and .csv")

    if args.gif:
        gif_dir = run_dir / "gifs"          # <run>/gifs next to <run>/checkpoints
        gif_dir.mkdir(exist_ok=True)
        for name in names:
            if args.gif_scenarios is not None and name not in args.gif_scenarios:
                continue
            i = names.index(name) * per
            fname = name if name != "custom" else "custom_" + "_".join(f"{v:g}" for v in scenarios[name])
            path = gif_dir / f"{args.env_name}_{fname}{'_' + tag if tag else ''}.gif"   # env- and tag-prefixed, so other scenes / checkpoints do not overwrite it
            render_gif(env, r["qpos"][i], r["qvel"][i], path, contact=r["contact"][i],
                       label=f"{name} cmd={tuple(scenarios[name])}", distance=args.gif_distance)
            print(f"wrote {path} (robot 0)")
    sys.stdout = console
    result_file.close()


if __name__ == "__main__":
    main()
