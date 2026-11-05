"""Joystick learning task"""

from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
from ml_collections import config_dict
from mujoco import mjx
from mujoco.mjx._src import math
import numpy as np

from mujoco_playground._src import mjx_env

import locomotion_rl.custom_robot.base as custom_base
from locomotion_rl.custom_robot.constants import consts

def default_config() -> config_dict.ConfigDict:
    return config_dict.create(
      ctrl_dt=0.02,
      sim_dt=0.004,
      episode_length=1000,
      Kp=consts.ACTUATOR_KP,
      Kd=consts.ACTUATOR_KV,
      action_repeat=1,
      action_scale=0.5,
      history_len=3,
      soft_joint_pos_limit_factor=0.95,
      noise_config=config_dict.create(
          level=1.0,  # Set to 0.0 to disable noise.
          scales=config_dict.create(
              joint_pos=consts.SERVO_BACKLASH,
              joint_vel=0.0,
              gyro=0.2,
              gravity=0.05,
              linvel=0.1,
          ),
      ),
      reward_config=config_dict.create(
          scales=config_dict.create(
              # Tracking.
              tracking_lin_vel=1.0,
              tracking_ang_vel=0.5,
              # Base reward.
              lin_vel_z=-0.5,
              ang_vel_xy=-0.05,
              orientation=-5.0,
              # Other.
              dof_pos_limits=-1.0,
              pose=0.1,
              # Other.
              termination=-1.0,
              stand_still=-0.1,
              # Regularization.
              # Note: These terms are much higher than go1 
              # to account for smaller scale actuators.
              torques=-0.001,
              action_rate=-0.01,
              energy=-0.001,
              # Feet.
              # Gait shape (Go1's terms). Off by default: they prescribe the gait instead of letting it emerge.
              feet_clearance=0.0,
              feet_height=0.0,
              feet_slip=0.0,
              feet_air_time=1.0,
          ),
          # Multiplies the energy weight: start * (end/start)^progress, progress = train_step / ramp_steps. Off at 1 -> 1.
          # ramp_steps counts steps of ONE env (total env steps / num_envs); the counter lives in state.info, which the
          # Brax auto-reset keeps across episodes (locomotion.py --energy_schedule fills these in).
          # initial_step: None starts finished (scale = end: eval envs, rollouts); 0 starts the ramp (training env).
          energy_schedule=config_dict.create(start=1.0, end=1.0, ramp_steps=0, initial_step=None),
          tracking_sigma=0.03,
          max_foot_height=0.02,       # m, target swing peak for feet_height / feet_clearance (Go1: 0.1 m on a ~0.3 m leg)
          air_time_threshold=0.15,     # s, swings shorter than this are penalised by feet_air_time
      ),
    #   disturb_config=config_dict.create(
    #       enable=False,
    #       velocity_kick=[0.0, 0.5],
    #       kick_durations=[0.05, 0.2],
    #       kick_wait_times=[1.0, 3.0],
    #   ),
      command_config=config_dict.create(
          # Uniform distribution for command amplitude.
          a=[0.5, 0.2, 1.0],
          # Probability of not zeroing out new command.
          b=[0.9, 0.25, 0.5],
      ),
      impl="warp",
      iterations=None,        # solver iteration limits; None keeps the value in the MJX xml (gen_custom_mjx.py)
      ls_iterations=None,
      naconmax=4 * 8192,
      ccd_iterations=None,    # None = 20 (base.py)
      naccdmax=None,          # warp GJK/EPA slots (heightfield-vs-cylinder feet only); None = naconmax. See params.py
      njmax=86,
  )

class Joystick(custom_base.CustomRobotEnv):
    def __init__(
        self,
        task: str = "flat_terrain",
        config: config_dict.ConfigDict = default_config(),
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
    ):
        super().__init__(
            xml_path=consts.task_to_xml(task).as_posix(),
            config=config,
            config_overrides=config_overrides
        )
        self._post_init()

    def _post_init(self) -> None:
        self._init_q = jp.array(self._mj_model.keyframe(consts.KEYFRAME).qpos)
        self._default_pose = jp.array(self._mj_model.keyframe(consts.KEYFRAME).qpos[7:])

        self._lowers, self._uppers = self.mj_model.jnt_range[1:].T
        self._soft_lowers = self._lowers * self._config.soft_joint_pos_limit_factor
        self._soft_uppers = self._uppers * self._config.soft_joint_pos_limit_factor

        self._torso_body_id = self._mj_model.body(consts.ROOT_BODY).id
        self._toros_mass = self.mj_model.body_subtreemass[self._torso_body_id]

        self._feet_site_id = np.array(
            [self.mj_model.site(name).id for name in consts.FEET_SITES]
        )
        # self._floor_geom_id
        # self._feet_geom_id

        foot_linvel_sensor_adr = []
        for sensor_name in consts.FEET_GLOBAL_LINVEL_SENSOR:
            sensor_id = self._mj_model.sensor(sensor_name).id
            sensor_adr = self._mj_model.sensor_adr[sensor_id]
            sensor_dim = self._mj_model.sensor_dim[sensor_id]
            foot_linvel_sensor_adr.append(
                list(range(sensor_adr, sensor_adr + sensor_dim))
            )
        self._foot_linvel_sensor_adr = jp.array(foot_linvel_sensor_adr)

        self._foot_touch_adr = jp.array(
            [self._mj_model.sensor_adr[self._mj_model.sensor(n).id] for n in consts.FEET_TOUCH_SENSORS]
        )

        self._cmd_a = jp.array(self._config.command_config.a)
        self._cmd_b = jp.array(self._config.command_config.b)

        self._joint_max_vel = consts.MAX_JOINT_VELOCITY
        self._init_terrain()

    def _init_terrain(self) -> None:
        """Heightfield scenes: a table of the highest terrain point under the robot's footprint, for the reset positions.

        Without it the robot would spawn inside a bump (or hang high above a dip, then drop). Flat scenes: no table, height 0.
        """
        self._terrain = None
        m = self._mj_model
        if not m.nhfield:
            return
        nrow, ncol = int(m.hfield_nrow[0]), int(m.hfield_ncol[0])
        rx, ry, elevation, _ = m.hfield_size[0]
        z = m.hfield_data[: nrow * ncol].reshape(nrow, ncol) * elevation + m.geom_pos[m.geom(consts.FLOOR_GEOM).id, 2]
        dx, dy = 2 * rx / (ncol - 1), 2 * ry / (nrow - 1)                       # column j is x = -rx + j dx, row i is y = -ry + i dy
        reach, foot = 0.5 + 0.2, 0.1                                              # reset offset range + margin; half the robot's length
        j0, i0 = int(round(rx / dx)), int(round(ry / dy))
        nj, ni, fj, fi = int(reach / dx), int(reach / dy), int(foot / dx), int(foot / dy)
        window = z[i0 - ni - fi: i0 + ni + fi + 1, j0 - nj - fj: j0 + nj + fj + 1]
        out = window.copy()
        for k in range(-fj, fj + 1):                                              # separable max filter over the footprint
            out = np.maximum(out, np.roll(window, k, axis=1))
        window = out.copy()
        for k in range(-fi, fi + 1):
            out = np.maximum(out, np.roll(window, k, axis=0))
        self._terrain = dict(
            height=jp.array(out[fi: -fi, fj: -fj]), dx=float(dx), dy=float(dy), nj=nj, ni=ni)

    def _terrain_height(self, x: jax.Array, y: jax.Array) -> jax.Array:
        """Highest terrain within the footprint around (x, y), for |x|, |y| <= 0.7 m."""
        if self._terrain is None:
            return jp.zeros(())
        t = self._terrain
        j = jp.clip(jp.round(x / t["dx"]).astype(jp.int32) + t["nj"], 0, t["height"].shape[1] - 1)
        i = jp.clip(jp.round(y / t["dy"]).astype(jp.int32) + t["ni"], 0, t["height"].shape[0] - 1)
        return t["height"][i, j]

    def reset(self, rng: jax.Array) -> mjx_env.State:
        qpos = self._init_q
        qvel = jp.zeros(self.mjx_model.nv)

        # Randomly translate (-0.5, 0.5)
        rng, key = jax.random.split(rng)
        dxy = jax.random.uniform(key, (2,), minval=-0.5, maxval=0.5)
        qpos = qpos.at[0:2].set(qpos[0:2] + dxy)
        qpos = qpos.at[2].add(self._terrain_height(dxy[0], dxy[1]) + 0.002)      # on rough terrain: above the highest bump under the robot

        # Randomly rotate (-PI, PI)
        rng, key = jax.random.split(rng)
        yaw = jax.random.uniform(key, (1,), minval=-3.1415, maxval=3.1415)
        quat = math.axis_angle_to_quat(jp.array([0, 0, 1]), yaw)
        new_quat = math.quat_mul(qpos[3:7], quat)
        qpos = qpos.at[3:7].set(new_quat)

        # Random velocity(?)
        rng, key = jax.random.split(rng)
        qvel = qvel.at[0:6].set(
            jax.random.uniform(key, (6,), minval=-0.5, maxval=0.5)
        )

        data = mjx_env.make_data(
            self.mj_model,
            qpos=qpos,
            qvel=qvel,
            ctrl=qpos[7:],
            impl=self.mjx_model.impl.value,
            naconmax=self._config.naconmax,
            naccdmax=self._config.naccdmax,
            njmax=self._config.njmax,
        )
        data = mjx.forward(self.mjx_model, data)

        rng, key1, key2 = jax.random.split(rng, 3)
        time_until_next_cmd = jax.random.exponential(key1) * 5.0
        steps_until_next_cmd = jp.round(time_until_next_cmd / self.dt).astype(
            jp.int32
        )
        cmd = jax.random.uniform(
            key2, shape=(3,), minval=-self._cmd_a, maxval=self._cmd_a
        )

        info = {
            "rng": rng,
            "command": cmd,
            "steps_until_next_cmd": steps_until_next_cmd,
            "last_act": jp.zeros(self.mjx_model.nu),
            "last_last_act": jp.zeros(self.mjx_model.nu),
            "last_contact": jp.zeros(4, dtype=bool),
            "feet_air_time": jp.zeros(4),
            "last_target" : self._default_pose,
            "swing_peak": jp.zeros(4),
            "train_step": jp.array(self._energy_ramp_initial_step(), dtype=jp.int32),
        }

        metrics = {}
        for k in self._config.reward_config.scales.keys():
            metrics[f"reward/{k}"] = jp.zeros(())
            metrics["swing_peak"] = jp.zeros(())
        metrics["energy_scale"] = jp.zeros(())

        obs = self._get_obs(data, info)
        reward, done = jp.zeros(2)
        return mjx_env.State(data, obs, reward, done, metrics, info)


    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        actuator_targets = self._default_pose + action * self._config.action_scale
        max_step = self._joint_max_vel * self._ctrl_dt
        clamped_targets = state.info["last_target"] + \
            (actuator_targets - state.info["last_target"]).clip(-max_step, max_step)
        state.info["last_target"] = clamped_targets

        data = mjx_env.step(
            self.mjx_model, state.data, clamped_targets, self.n_substeps
        )

        contact = jp.array([
            data.sensordata[self._mj_model.sensor_adr[sensorid]] > 0
            for sensorid in self._feet_floor_found_sensor
        ])
        contact_filt = contact | state.info["last_contact"]     # bridges a single-step contact flicker
        first_contact = (state.info["feet_air_time"] > 0.0) * contact_filt
        state.info["feet_air_time"] += self.dt
        p_f = data.site_xpos[self._feet_site_id]
        p_fz = p_f[..., -1]
        state.info["swing_peak"] = jp.maximum(state.info["swing_peak"], p_fz)

        obs = self._get_obs(data, state.info)
        done = self._get_termination(data)

        rewards = self._get_reward(
            data, action, state.info, state.metrics, done, first_contact, contact
        )
        rewards = {
            k: v * self._config.reward_config.scales[k] for k, v in rewards.items()
        }
        energy_scale = self._energy_scale(state.info["train_step"])
        rewards["energy"] = rewards["energy"] * energy_scale
        state.info["train_step"] = state.info["train_step"] + 1
        reward = jp.clip(sum(rewards.values()) * self.dt, 0.0, 10000.0)

        state.info["last_last_act"] = state.info["last_act"]
        state.info["last_act"] = action
        state.info["steps_until_next_cmd"] -= 1
        state.info["rng"], key1, key2 = jax.random.split(state.info["rng"], 3)
        state.info["command"] = jp.where(
            state.info["steps_until_next_cmd"] <= 0,
            self.sample_command(key1, state.info["command"]),
            state.info["command"],
        )
        state.info["steps_until_next_cmd"] = jp.where(
            done | (state.info["steps_until_next_cmd"] <= 0),
            jp.round(jax.random.exponential(key2) * 5.0 / self.dt).astype(jp.int32),
            state.info["steps_until_next_cmd"],
        )

        state.info["feet_air_time"] *= ~contact
        state.info["last_contact"] = contact
        state.info["swing_peak"] *= ~contact
        for k, v in rewards.items():
            state.metrics[f"reward/{k}"] = v
        state.metrics["swing_peak"] = jp.mean(state.info["swing_peak"])
        state.metrics["energy_scale"] = energy_scale

        done = done.astype(reward.dtype)
        state = state.replace(
            data=data, obs=obs, reward=reward, done=done
        )
        return state

    def _energy_ramp_initial_step(self) -> int:
        sch = self._config.reward_config.energy_schedule
        return sch.ramp_steps if sch.initial_step is None else sch.initial_step

    def _energy_scale(self, train_step: jax.Array) -> jax.Array:
        """Geometric ramp of the energy weight from `start` to `end` (a straight line in log space)."""
        sch = self._config.reward_config.energy_schedule
        progress = jp.clip(train_step / max(sch.ramp_steps, 1), 0.0, 1.0)
        return sch.start * (sch.end / sch.start) ** progress

    def _get_termination(self, data: mjx.Data) -> jax.Array:
        fell_over = self.get_upvector(data)[-1] < 0.0
        fell_off = data.qpos[2] < -0.1                       # walked off the edge of the heightfield
        return fell_over | fell_off

    def _get_obs(
        self, data: mjx.Data, info: dict[str, Any]
    ) -> Dict[str, jax.Array]:
        gyro = self.get_gyro(data)
        info["rng"], noise_rng = jax.random.split(info["rng"])
        noisy_gyro = (
            gyro
            + (2 * jax.random.uniform(noise_rng, shape=gyro.shape) - 1)
            * self._config.noise_config.level
            * self._config.noise_config.scales.gyro
        )

        gravity = self.get_gravity(data)
        info["rng"], noise_rng = jax.random.split(info["rng"])
        noisy_gravity = (
            gravity
            + (2 * jax.random.uniform(noise_rng, shape=gravity.shape) - 1)
            * self._config.noise_config.level
            * self._config.noise_config.scales.gravity
        )

        joint_angles = data.qpos[7:]
        info["rng"], noise_rng = jax.random.split(info["rng"])
        noisy_joint_angles = (
            joint_angles
            + (2 * jax.random.uniform(noise_rng, shape=joint_angles.shape) - 1)
            * self._config.noise_config.level
            * self._config.noise_config.scales.joint_pos
        )

        joint_vel = data.qvel[6:]
        info["rng"], noise_rng = jax.random.split(info["rng"])
        noisy_joint_vel = (
            joint_vel
            + (2 * jax.random.uniform(noise_rng, shape=joint_vel.shape) - 1)
            * self._config.noise_config.level
            * self._config.noise_config.scales.joint_vel
        )

        state = jp.hstack([
            noisy_gyro,
            noisy_gravity,
            noisy_joint_angles - self._default_pose,
            noisy_joint_vel,
            info["last_act"],
            info["last_last_act"],
            info["command"],
        ])

        accel = self.get_accelerometer(data)
        linvel = self.get_local_linvel(data)
        angvel = self.get_global_angvel(data)
        feet_vel = data.sensordata[self._foot_linvel_sensor_adr].ravel()

        privileged_state = jp.hstack([
            state,
            gyro,
            accel,
            gravity,
            linvel,
            angvel,
            joint_angles - self._default_pose,
            joint_vel,
            data.actuator_force,
            info["last_contact"],
            feet_vel,
            data.xfrc_applied[self._torso_body_id, :3],
        ])

        return {
            "state": state,
            "privileged_state": privileged_state,
        }

    def _get_reward(
        self,
        data: mjx.Data,
        action: jax.Array,
        info: dict[str, Any],
        metrics: dict[str, Any],
        done: jax.Array,
        first_contact: jax.Array,
        contact: jax.Array,
    ) -> dict[str, jax.Array]:
        # TODO: Original did not use metrics even though they were generated?
        del metrics 
        
        return {
            "tracking_lin_vel": self._reward_tracking_lin_vel(
                info["command"], self.get_local_linvel(data)
            ),
            "tracking_ang_vel": self._reward_tracking_ang_vel(
                info["command"], self.get_gyro(data)
            ),
            "lin_vel_z": self._cost_lin_vel_z(self.get_global_linvel(data)),
            "ang_vel_xy": self._cost_ang_vel_xy(self.get_global_angvel(data)),
            "orientation": self._cost_orientation(self.get_upvector(data)),
            "stand_still": self._cost_stand_still(info["command"], data.qpos[7:]),
            "termination": self._cost_termination(done),
            "pose" : self._reward_pose(data.qpos[7:]),
            "torques": self._cost_torques(data.actuator_force),
            "action_rate": self._cost_action_rate(
                action, info["last_act"], info["last_last_act"]
            ),
            "energy": self._cost_energy(data.qvel[6:], data.actuator_force),
            "feet_slip": self._cost_feet_slip(data, contact, info),
            "dof_pos_limits": self._cost_joint_pos_limits(data.qpos[7:]),
            "feet_clearance": self._cost_feet_clearance(data),
            "feet_height": self._cost_feet_height(info["swing_peak"], first_contact, info),
            "feet_air_time": self._reward_feet_air_time(info["feet_air_time"], first_contact, info["command"]),
            "dof_pos_limits": self._cost_joint_pos_limits(data.qpos[7:]),
        }

    # Tracking rewards

    def _reward_tracking_lin_vel(
        self,
        commands: jax.Array,
        local_vel: jax.Array,
    ) -> jax.Array:
        # Track linvel in xy
        lin_vel_xy_error = jp.sum(jp.square(commands[:2] - local_vel[:2]))
        return jp.exp(-lin_vel_xy_error / self._config.reward_config.tracking_sigma)

    def _reward_tracking_ang_vel(
        self,
        commands: jax.Array,
        ang_vel: jax.Array,
    ) -> jax.Array:
        # Track yaw commands
        yaw_vel_error = jp.square(commands[2] - ang_vel[2])
        return jp.exp(-yaw_vel_error / self._config.reward_config.tracking_sigma)

    # State rewards

    def _cost_lin_vel_z(self, global_linvel) -> jax.Array:
        return jp.square(global_linvel[2])

    def _cost_ang_vel_xy(self, global_angvel) -> jax.Array:
        return jp.sum(jp.square(global_angvel[:2]))

    def _cost_orientation(self, torso_zaxis: jax.Array) -> jax.Array:
        return jp.sum(jp.square(torso_zaxis[:2]))

    # Energy-based rewards

    def _cost_torques(self, torques: jax.Array) -> jax.Array:
        return jp.sqrt(jp.sum(jp.square(torques / consts.TORQUE_LIMIT)))

    def _cost_energy(
        self, qvel: jax.Array, qfrc_actuator: jax.Array
    ) -> jax.Array:
        return jp.sum(jp.abs(qvel) * jp.abs(qfrc_actuator / consts.TORQUE_LIMIT))

    def _cost_action_rate(
        self, act: jax.Array, last_act: jax.Array, last_last_act: jax.Array
    ) -> jax.Array:
        return jp.sum(jp.square(act - last_act))

    def _reward_pose(self, qpos: jax.Array) -> jax.Array:
        # Stay close to the default pose.
        weight = jp.array([1.0, 1.0, 0.1] * 4)
        return jp.exp(-jp.sum(jp.square(qpos - self._default_pose) * weight))

    def _cost_stand_still(
        self,
        commands: jax.Array,
        qpos: jax.Array,
    ) -> jax.Array:
        cmd_norm = jp.linalg.norm(commands)
        return jp.sum(jp.abs(qpos - self._default_pose)) * (cmd_norm < 0.01)

    def _cost_termination(self, done: jax.Array) -> jax.Array:
        # Penalize early termination.
        return done

    def _cost_feet_slip(
        self, data: mjx.Data, contact: jax.Array, info: dict[str, Any]
    ) -> jax.Array:
        cmd_norm = jp.linalg.norm(info["command"])
        feet_vel = data.sensordata[self._foot_linvel_sensor_adr]
        vel_xy = feet_vel[..., :2]
        vel_xy_norm_sq = jp.sum(jp.square(vel_xy), axis=-1)
        return jp.sum(vel_xy_norm_sq * contact) * (cmd_norm > 0.01)

    def _cost_joint_pos_limits(self, qpos: jax.Array) -> jax.Array:
        out_of_limits = -jp.clip(qpos - self._soft_lowers, None, 0.0)
        out_of_limits += jp.clip(qpos - self._soft_uppers, 0.0, None)
        return jp.sum(out_of_limits)

    def _cost_feet_clearance(self, data: mjx.Data) -> jax.Array:
        feet_vel = data.sensordata[self._foot_linvel_sensor_adr]
        vel_norm = jp.sqrt(jp.linalg.norm(feet_vel[..., :2], axis=-1))
        foot_z = data.site_xpos[self._feet_site_id][..., -1]
        delta = jp.abs(foot_z - self._config.reward_config.max_foot_height)
        return jp.sum(delta * vel_norm)

    def _cost_feet_height(self, swing_peak: jax.Array, first_contact: jax.Array, info: dict[str, Any]) -> jax.Array:
        cmd_norm = jp.linalg.norm(info["command"])
        error = swing_peak / self._config.reward_config.max_foot_height - 1.0
        return jp.sum(jp.square(error) * first_contact) * (cmd_norm > 0.01)

    def _reward_feet_air_time(self, air_time: jax.Array, first_contact: jax.Array, commands: jax.Array) -> jax.Array:
        cmd_norm = jp.linalg.norm(commands)
        rew = jp.sum((air_time - self._config.reward_config.air_time_threshold) * first_contact)
        return rew * (cmd_norm > 0.01)

    def sample_command(self, rng: jax.Array, x_k: jax.Array) -> jax.Array:
        rng, y_rng, w_rng, z_rng = jax.random.split(rng, 4)
        y_k = jax.random.uniform(
            y_rng, shape=(3,), minval=-self._cmd_a, maxval=self._cmd_a
        )
        z_k = jax.random.bernoulli(z_rng, self._cmd_b, shape=(3,))
        w_k = jax.random.bernoulli(w_rng, 0.5, shape=(3,))
        x_kp1 = x_k - w_k * (x_k - y_k * z_k)
        return x_kp1