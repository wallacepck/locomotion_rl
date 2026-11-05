import sys
from typing import Any, Dict, Optional, Union

from etils import epath
import jax
import jax.numpy as jp
from ml_collections import config_dict
import mujoco
from mujoco import mjx

from mujoco_playground._src import mjx_env

from locomotion_rl.custom_robot.constants import consts

# mujoco_warp overflow warnings (mjw.OverflowType names) that are switched off. ITERATIONS is "solver iteration limit
# reached": it fires in about 0.5 ppm of the steps (opt.iterations = 10), so it only floods the logs.
SUPPRESSED_OVERFLOWS = ("ITERATIONS",)
_announced = False


def _suppress_overflow_warnings(mjx_model: mjx.Model) -> mjx.Model:
    """Clears the SUPPRESSED_OVERFLOWS bits of the warp model's opt.warn_overflow, and says so once per process."""
    global _announced
    if not SUPPRESSED_OVERFLOWS:
        return mjx_model
    from mujoco.mjx.third_party.mujoco_warp import OverflowType
    mask = 0
    for name in SUPPRESSED_OVERFLOWS:
        mask |= int(OverflowType[name])
    warn = mjx_model.opt._impl.warn_overflow
    if not _announced:
        _announced = True
        print(f"WARNING: mujoco_warp overflow warnings suppressed: {', '.join(SUPPRESSED_OVERFLOWS)} "
              f"(custom_robot/base.py SUPPRESSED_OVERFLOWS)", file=sys.stderr)
    return mjx_model.tree_replace({"opt._impl.warn_overflow": warn & ~mask})


def get_assets() -> Dict[str, bytes]:
    assets = {}
    mjx_env.update_assets(assets, consts.XML_DIR, "*.xml")
    mjx_env.update_assets(assets, consts.XML_DIR / "assets")      # meshes, looked up by bare file name
    return assets

class CustomRobotEnv(mjx_env.MjxEnv):
    def __init__(
        self, 
        xml_path: str, 
        config: config_dict.ConfigDict,
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
    ) -> None:
        super().__init__(config, config_overrides)

        self._model_assets = get_assets()
        self._mj_model = mujoco.MjModel.from_xml_string(
            epath.Path(xml_path).read_text(), assets=self._model_assets
        )
        # Joint order contract (constants.py): actuator i drives dof 6 + i, so qpos[7:], qvel[6:], ctrl and actions share one order.
        mj = self._mj_model
        assert [mj.actuator(i).name for i in range(mj.nu)] == list(consts.ACTUATOR_NAMES), "actuator order differs from consts"
        assert all(mj.jnt_dofadr[mj.actuator_trnid[i, 0]] == consts.NUM_ROOT_DOF + i for i in range(mj.nu)), "actuators not in joint order"

        self._mj_model.opt.timestep = self._config.sim_dt
        self._mj_model.opt.ccd_iterations = self._config.get("ccd_iterations") or 20    # EPA iterations; warp sizes its scratch buffers by it
        if self._config.get("iterations") is not None:
            self._mj_model.opt.iterations = self._config.iterations
        if self._config.get("ls_iterations") is not None:
            self._mj_model.opt.ls_iterations = self._config.ls_iterations

        # self._mj_model.dof_damping[6:] = config.Kd
        self._mj_model.actuator_gainprm[:, 0] = config.Kp
        self._mj_model.actuator_biasprm[:, 1] = -config.Kp     # position term:   tau = kp (target - q) - kv qdot
        self._mj_model.actuator_biasprm[:, 2] = -config.Kd     # velocity term (the actuator's kv; passive joint damping is left alone)

        self._mjx_model = mjx.put_model(
            self._mj_model, impl=self._config.impl
        )
        if self._config.impl == "warp":
            self._mjx_model = _suppress_overflow_warnings(self._mjx_model)
        self._xml_path = xml_path
        self._imu_site_id = self._mj_model.site("imu").id

        # sensor ids (not names) of the per-foot floor contact sensors; step() reads sensordata[sensor_adr[id]] > 0
        self._feet_floor_found_sensor = [self._mj_model.sensor(n).id for n in consts.FEET_FLOOR_FOUND_SENSOR]

    def get_upvector(self, data: mjx.Data) -> jax.Array:
        return mjx_env.get_sensor_data(self.mj_model, data, consts.UPVECTOR_SENSOR)

    def get_gravity(self, data: mjx.Data) -> jax.Array:
        return data.site_xmat[self._imu_site_id].T @ jp.array([0, 0, -1])

    def get_global_linvel(self, data: mjx.Data) -> jax.Array:
        return mjx_env.get_sensor_data(
            self.mj_model, data, consts.GLOBAL_LINVEL_SENSOR
        )

    def get_global_angvel(self, data: mjx.Data) -> jax.Array:
        return mjx_env.get_sensor_data(
            self.mj_model, data, consts.GLOBAL_ANGVEL_SENSOR
        )

    def get_local_linvel(self, data: mjx.Data) -> jax.Array:
        return mjx_env.get_sensor_data(
            self.mj_model, data, consts.LOCAL_LINVEL_SENSOR
        )

    def get_accelerometer(self, data: mjx.Data) -> jax.Array:
        return mjx_env.get_sensor_data(
            self.mj_model, data, consts.ACCELEROMETER_SENSOR
        )

    def get_gyro(self, data: mjx.Data) -> jax.Array:
        return mjx_env.get_sensor_data(self.mj_model, data, consts.GYRO_SENSOR)

    def get_feet_pos(self, data: mjx.Data) -> jax.Array:
        return jp.vstack([
            mjx_env.get_sensor_data(self.mj_model, data, sensor_name)
            for sensor_name in consts.FEET_POS_SENSOR
        ])

    @property
    def xml_path(self) -> str:
        return self._xml_path

    @property
    def action_size(self) -> int:
        return self._mjx_model.nu

    @property
    def mj_model(self) -> mujoco.MjModel:
        return self._mj_model

    @property
    def mjx_model(self) -> mjx.Model:
        return self._mjx_model
