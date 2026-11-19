# Project setup: quadruped RL + CHAMP sim-to-sim workspace

How to rebuild the whole workspace and run each task. The workspace holds a custom quadruped (SC-009 servos) with two controllers,
the CHAMP model-based controller and a PPO policy trained here, and the tools to train, roll out, transfer and race them.

Versions below are the ones the project was built and tested with. Hardware figures are recommended ranges, not hard limits.

```
chvmp_ws/                      <- the colcon workspace (WS below)
├── venv/                      <- venv #1: CPU MuJoCo for the ROS simulation node          (requirements-sim.txt)
└── src/
    ├── champ/                 <- CHAMP controller + champ_mujoco (CPU MuJoCo bridge)
    ├── custom_robot_desc/     <- the robot's URDF/xacro/meshes (ROS package name: urdf_robot_description)
    ├── champ_transfer/        <- sim-to-sim transfer, race, trials, post-edit video tools
    └── locomotion_rl/         <- Brax PPO + MJX/Warp training
        ├── venv/              <- venv #2: JAX / Brax / Warp / MuJoCo 3.16                 (project/requirements.txt)
        └── project/           <- this folder
```

---

## 1. System requirements

| | |
|---|---|
| OS | Ubuntu 24.04 |
| ROS 2 | **Jazzy** (`/opt/ros/jazzy`) |
| Python | 3.12 (Ubuntu 24.04's `python3`; both venvs are built from it) |
| GPU | NVIDIA, for training and the GPU rollouts (see below). The CPU simulation, the course race and the video tools do not need it |
| Disk | a few GB for the venvs (the CUDA wheels dominate) plus `logs/` and `results/` |
| CPU / RAM | 4+ cores and 16 GB RAM recommended (the CPU simulation and several parallel race runs benefit from more cores) |

### GPU / CUDA (Warp, JAX, Brax)

- Training and `rollout.py` run MuJoCo MJX on the **Warp** backend (`impl="warp"`, the default) with JAX/Brax PPO on top.
  The pinned wheels are **CUDA 13** builds: `jax-cuda13-plugin`, `jax-cuda13-pjrt`, `nvidia-*-cu13`, `warp-lang 1.18`.
  The CUDA toolkit does **not** need to be installed system-wide: the `nvidia-cuda-*` pip wheels bring the runtime and nvcc.
- **Driver:** a recent NVIDIA driver that supports CUDA 13 (NVIDIA's CUDA 13.0 release notes: 580 or newer). Check with `nvidia-smi`;
  Warp prints the device it picked when a script starts.
- **GPU generation:** Turing (compute capability 7.5) or newer is recommended, since CUDA 13 dropped the older architectures
  (see the JAX and Warp install pages for the exact support matrix of the versions you pin).
- **GPU memory:** 8 GB or more is recommended. Training at the default 8192 envs peaks at roughly 1.5 to 2.5 GB, so a 44M-step run fits
  on a mid-range 8 GB card and two jobs can share it. With less memory lower `--num_envs`. Check `nvidia-smi` before starting a job:
  an out-of-memory error mid-run costs more than the check.
- **Run time:** a 44M-step run takes on the order of 5 to 20 minutes on a mid-range GPU, depending on the terrain (flat is fastest,
  the fine 5 mm block heightfield slowest). Long runs (200M steps) are better sent to a cluster GPU (`job.sh`, section 6.1).
- On a machine with multiple GPUs the MuJoCo **window** (GLFW) may default to the integrated GPU. The launch files and
  `compose_race --view` already set `__NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia`; offscreen EGL rendering uses NVIDIA by itself.

---

## 2. Get the code (external links)

The workspace is the 4 repositories below. Clone into `WS/src` (any location works; `~/chvmp_ws` is used below).

```bash
mkdir -p ~/chvmp_ws/src && cd ~/chvmp_ws/src

# CHAMP controller (ROS 2 port). The project's fork carries champ_mujoco (the CPU MuJoCo bridge) and the custom-robot config.
git clone https://github.com/wallacepck/champ.git            # upstream: https://github.com/chvmp/champ  (branch ros2)

# the robot description (the repo is custom_robot_desc, its ROS package is urdf_robot_description)
git clone https://github.com/wallacepck/custom_robot_desc.git

# this project's packages
git clone https://github.com/wallacepck/champ_transfer.git   # transfer, race, trials, compose_race
git clone git@github.com:wallacepck/locomotion_rl.git        # RL training (SSH remote; use the https URL if you have no key)
```

Further external references: MuJoCo (<https://github.com/google-deepmind/mujoco>), MuJoCo Playground
(<https://github.com/google-deepmind/mujoco_playground>, installed from PyPI as `playground`), MuJoCo Warp
(<https://github.com/google-deepmind/mujoco_warp>, shipped inside `mujoco-mjx`), Brax (<https://github.com/google/brax>).

---

## 3. ROS 2 Jazzy

```bash
# ROS itself: follow https://docs.ros.org/en/jazzy/Installation/Ubuntu-Install-Debs.html, then
sudo apt install ros-jazzy-desktop python3-colcon-common-extensions python3-rosdep python3-venv python3-pip ffmpeg
sudo rosdep init && rosdep update
```

- Packages the workspace pulls in beyond `ros-jazzy-desktop` (from the packages' `package.xml` and launch files):
  `xacro`, `robot_state_publisher`, `joint_state_publisher`, `rviz2`, plus whatever CHAMP lists
  (`nav2_bringup`, `slam_toolbox`, `robot_localization`, `ros_gz_sim` are not needed for the RL work).
  Let `rosdep` resolve them:
  ```bash
  cd ~/chvmp_ws
  source /opt/ros/jazzy/setup.bash
  rosdep install --from-paths src --ignore-src -r -y
  ```
- `ffmpeg` is only for MP4 output of `compose_race` (it falls back to GIF without it).

---

## 4. The two Python venvs

The ROS nodes run in ROS's own python, which has no MuJoCo, and the RL stack needs a very different (and heavy) set of wheels.
So there are two separate venvs, both `include-system-site-packages = false`:

| venv | location | used by | requirements |
|---|---|---|---|
| **RL venv** | `src/locomotion_rl/venv` | training, `rollout`, `export_policy`, `compare_models`, `compose_race` | `project/requirements.txt` (94 packages, exact `pip freeze`) |
| **sim venv** | `venv` (workspace root) | the CPU `champ_mujoco` / `champ_transfer` simulation node; it imports `mujoco` from here (appended to `sys.path`, so ROS keeps its own numpy) | `project/requirements-sim.txt` (10 packages) |

```bash
cd ~/chvmp_ws/src/locomotion_rl
python3 -m venv venv
venv/bin/pip install --upgrade pip
venv/bin/pip install -r project/requirements.txt          # downloads the CUDA 13 wheels, several GB

cd ~/chvmp_ws
python3 -m venv venv
venv/bin/pip install -r src/locomotion_rl/project/requirements-sim.txt
```

Key pins of `requirements.txt` (full list in the file): `jax / jaxlib 0.11.2` + `jax-cuda13-plugin`, `brax 0.14.2`,
`mujoco / mujoco-mjx 3.16.0`, `warp-lang 1.18.0`, `playground 0.2.0` (imports as `mujoco_playground`), `flax 0.12.10`,
`orbax-checkpoint 0.12.7`, `numpy 2.5.4`, `scipy 1.18.1`, `matplotlib 3.11.2`, `pillow 13.0.0`.

Notes:
- The RL venv is **not** a ROS venv. After `source /opt/ros/jazzy/setup.bash` the ROS python packages (including its older numpy)
  are on `PYTHONPATH` and can leak into a venv python. For training and rollouts use a shell where ROS has not been sourced
  (or prefix `env -u PYTHONPATH -u AMENT_PREFIX_PATH`, then set `PYTHONPATH=$PWD` again). The commands that need both ROS and the
  RL venv python (`export_policy`, `compare_models`) say so.
- The sim venv has MuJoCo **3.14** and the RL venv **3.16**: the CPU simulation and the RL environment use different
  MuJoCo versions. That is how the transfer results were produced; `champ_transfer/compare_models` checks the two models still agree.
- Regenerate the lists after a change (do it in a shell without ROS sourced, or the ROS python packages leak into the freeze):
  `venv/bin/pip freeze > project/requirements.txt`.

---

## 5. Build and source

```bash
cd ~/chvmp_ws
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install                              # everything; or one package:
colcon build --symlink-install --packages-select champ_transfer
source install/setup.bash                                   # in every new terminal that uses ros2 / the champ_* tools
```

- `--symlink-install` matters: it lets the launch files, `gait.yaml`, scripts and exported policies be edited without rebuilding.
  A **new** file (for example a policy `.npz` from `export_policy`) still needs one `colcon build` so it is installed.
- `colcon list` should show `champ*`, `champ_mujoco`, `champ_transfer`, `locomotion_rl` and `urdf_robot_description` among its packages. `build/ install/ log/` are generated.
- The RL commands below do **not** need the colcon build or sourcing. Only the ROS/CHAMP ones do.

---

## 6. Tasks

All RL commands run from `src/locomotion_rl` with `PYTHONPATH=$PWD` (or `.`) and the venv python; check `nvidia-smi` first.

```bash
cd ~/chvmp_ws/src/locomotion_rl
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
export PYTHONPATH=$PWD
```

### 6.1 Train

```bash
# 44M-step run (workspace-sized): output logs/<env>-<timestamp>/{checkpoints,config.json,metrics.csv,reward.png}
venv/bin/python -u -m locomotion_rl.locomotion --env_name CustomRobotJoystickRoughTerrain --num_timesteps 44000000 --logdir logs
#   --env_name CustomRobotJoystickFlatTerrain | ...RoughTerrain | ...BlockTerrain
#   --num_envs 8192 (default)  --num_evals 10  --seed N  --impl warp|jax  --no_randomization
#   --env_override reward_config.tracking_sigma=0.03        (any env config key, repeatable)
#   --energy_schedule 1 10                                   (ramp the energy weight)
```

Rename or move a run folder to give it a name (`logs/Sluurm`, `logs/FinalGait`, ...). `config.json` inside it records the exact
env and PPO settings, and every tool below reads it. Reward terms and their weights are in `custom_robot/joystick.py`
(`default_config`); PPO settings in `custom_robot/params.py`.

Cluster (200M-step runs): `job.sh` is a Slurm script (`sbatch job.sh`, `ENV_NAME=... sbatch job.sh`, extra args go to `locomotion.py`).
Replace the `XXX` placeholders (`PKG_DIR`, mail user) and build the venv on the cluster from `project/requirements.txt` first.

### 6.2 Plot the training curves

```bash
venv/bin/python -m locomotion_rl.plot_rewards logs/Sluurm [logs/other ...] [--terms pose energy tracking_lin_vel] [--per_step] [--out file.png] [--list]
```

### 6.3 GPU rollout (MJX, the training simulator)

```bash
venv/bin/python -m locomotion_rl.rollout --checkpoint logs/Sluurm --env_name CustomRobotJoystickRoughTerrain
#   --checkpoint accepts a run dir (newest step), a checkpoints dir or one step dir: logs/Sluurm/checkpoints/000045875200
#   --gif [--gif_scenarios stand forward turn]   GIFs of robot 0 into <run>/gifs/ (only the named scenarios; the table still covers all)
#   --scenarios forward turn   --command 0.3 0 0   --seconds 10   --envs_per_scenario 16   --randomize   --tag NAME
```

Tables are printed and also written to `<run>/<env_name>_rollout[_<tag>].txt/.csv`. Compile several runs / scenes side by side:

```bash
venv/bin/python -m locomotion_rl.compile_rollouts logs/Sluurm logs/FinalGait --env CustomRobotJoystickRoughTerrain --scenarios forward turn
#   --matrix cot   a runs x evaluation-environments grid ("trained on A, evaluated on B");  --list shows the columns;  --out table.csv
```

### 6.4 Transfer rollout (the policy in champ_mujoco, CPU MuJoCo through ROS)

Needs the build (section 5) and both venvs. Details and the CHAMP-vs-RL comparison: `src/champ_transfer/README.md`.

```bash
cd ~/chvmp_ws/src/locomotion_rl
# 1. export a checkpoint to a numpy-only policy (verified against the jax policy), 2. install it
PYTHONPATH=.:../champ_transfer venv/bin/python -m champ_transfer.export_policy logs/Sluurm/checkpoints/000045875200 --out ../champ_transfer/policies/Sluurm_45M.npz
cd ~/chvmp_ws && colcon build --symlink-install --packages-select champ_transfer

# 3. evaluate headless: the same scenarios and metric code as rollout.py -> src/champ_transfer/results/<label>/
src/champ_transfer/scripts/evaluate.sh rl Sluurm_45M            # label = the exported policy's name
src/champ_transfer/scripts/evaluate.sh champ CHAMP               # the CHAMP controller
TERRAIN=rough src/champ_transfer/scripts/evaluate.sh rl Sluurm   # on the RL fractal terrain (files ChampMujocoRough_rollout.*)

# model parity check (CHAMP's MJCF vs the RL model, no GPU)
cd src/locomotion_rl && PYTHONPATH=.:../champ_transfer venv/bin/python -m champ_transfer.compare_models
```

Watch instead of headless: `ros2 launch champ_transfer rl.launch.py policy:=Sluurm` (or `champ.launch.py`) and publish to `/cmd_vel`.
The CHAMP-only simulator is `ros2 launch champ_mujoco mujoco.launch.py robot:=custom [gui:=false]`.

### 6.5 Course race (100 m dirt course, PD line follower)

```bash
cd ~/chvmp_ws && source /opt/ros/jazzy/setup.bash && source install/setup.bash
# live window, one controller
ros2 launch champ_transfer rl.launch.py   policy:=Sluurm course:=true follower:=true
ros2 launch champ_transfer champ.launch.py             course:=true follower:=true
# headless to the finish line (GUI=1 for the window); V_MAX speed command, LANE metres from the centre line (+ outward), SEED terrain seed
src/champ_transfer/scripts/race.sh rl Sluurm
V_MAX=0.3 LANE=0.4 src/champ_transfer/scripts/race.sh champ video/CHAMP
# 10 terrain seeds x (CHAMP 0.5, Sluurm 0.5, CHAMP 0.3 m/s), 5 in parallel, then statistics -> results/race/trials/{trials,summary}.csv
ros2 run champ_transfer race_trials [--seeds 10] [--parallel 5]
ros2 run champ_transfer race_stats
```

A run ends when it finishes, falls (tilt > 70 deg), makes < 0.5 m of progress in 20 s, or after 600 s. `race.sh` prints a health
line every 60 s (also in `<run>/health.log`). Each run writes `race.npz`, `race.txt`, `result.json` under `results/race/<label>/`.

Post-edit video of two recorded runs in one scene (RL venv python, no simulation):

```bash
cd ~/chvmp_ws/src/locomotion_rl
R=../champ_transfer/results/race/video
PYTHONPATH=../champ_transfer venv/bin/python -m champ_transfer.compose_race --view $R/CHAMP/race.npz $R/Sluurm/race.npz        # live window, to film
PYTHONPATH=../champ_transfer venv/bin/python -m champ_transfer.compose_race --out race.mp4 $R/CHAMP/race.npz $R/Sluurm/race.npz  # .gif or .mp4
#   --first-person Sluurm   camera on that robot (label, red/blue, 1/2);   --camera chase|overview|free;   --speed N;   --size W H
```

---

## 7. Where things are

| | |
|---|---|
| `locomotion_rl/custom_robot/joystick.py` | the environment: observations, commands, reward terms and their weights |
| `locomotion_rl/custom_robot/params.py` | PPO hyper-parameters |
| `locomotion_rl/custom_robot/terrain.py` | fractal and block terrain generators |
| `locomotion_rl/custom_robot/xmls/` | generated MJX scenes (flat / rough / block) |
| `logs/<run>/` | checkpoints, `config.json`, `metrics.csv`, rollout tables, `gifs/` |
| `champ_transfer/results/` | transfer evaluations, race records, trial statistics |
| `champ_transfer/reports/` | course previews and plots |
| CHAMP's `gait.yaml` | CHAMP's gait limits (`max_linear_velocity_x` was raised from 0.3 to 0.5 m/s for the race) |
