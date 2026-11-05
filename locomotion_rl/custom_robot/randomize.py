"""Domain randomisation for the custom robot (counterpart of mujoco_playground's go1 randomize.py).

Called once per training run by the Brax wrapper as `randomization_fn(mjx_model) -> (batched_model, in_axes)`: it draws
one set of model parameters per environment, and each environment keeps its set for the whole run (they are NOT
resampled at episode reset). So num_envs is also the number of distinct robots the policy sees.

What differs from Go1, and why:
  * The robot is 0.65 kg, not 12 kg, so absolute offsets (CoM jitter, added torso mass) are scaled down to match.
  * Go1 only randomises a torque-limited model; here the actuator itself is the main sim-to-real gap, so the position
    gain, the velocity gain, the torque limit (battery sag) and the passive joint damping are randomised too.
  * qpos0 is the joint zero: shifting it models servo calibration/horn-mounting offset (the policy sees angles
    relative to the default pose, so an offset it cannot observe has to be robust to).

All ranges below are guesses until the servos are identified on hardware. Tighten or widen them in the RANGES table,
nowhere else.
"""
import functools

import jax
import mujoco
from mujoco import mjx

from locomotion_rl.custom_robot.base import get_assets
from locomotion_rl.custom_robot.constants import consts

# (low, high); scale entries multiply the nominal value, offset entries add to it.
RANGES = dict(
    floor_friction=(0.4, 1.0),          # absolute coefficient (nominal 0.8)
    frictionloss_scale=(0.7, 1.3),      # gearbox/static friction is the least-known servo parameter
    damping_scale=(0.7, 1.3),
    armature_scale=(0.8, 1.4),          # reflected rotor inertia
    kp_scale=(0.8, 1.2),                # servo P gain register quantisation / voltage dependence
    kv_scale=(0.8, 1.2),
    torque_limit_scale=(0.8, 1.0),      # only ever lower than nominal: stall torque is the datasheet maximum
    link_mass_scale=(0.9, 1.1),
    torso_mass_offset=(-0.05, 0.05),    # kg (battery, cabling); ~8 % of the 0.65 kg total
    torso_com_offset=(-0.01, 0.01),     # m, each axis
    joint_zero_offset=(-0.03, 0.03),    # rad
)


@functools.lru_cache(maxsize=None)
def _ids() -> tuple[int, int]:
    """(floor geom id, torso body id), looked up by name from the same scene the environment loads."""
    mj = mujoco.MjModel.from_xml_string(consts.FLAT_TERRAIN_XML.read_text(), assets=get_assets())
    return mj.geom(consts.FLOOR_GEOM).id, mj.body(consts.ROOT_BODY).id


def domain_randomize(model: mjx.Model, rng: jax.Array):
    floor_id, torso_id = _ids()
    dof = slice(consts.NUM_ROOT_DOF, None)       # joint dofs: qvel[6:] == actuator order
    nj = consts.NUM_JOINTS
    r = RANGES

    @jax.vmap
    def rand_dynamics(rng):
        def u(key, rng_pair, shape=()):
            return jax.random.uniform(key, shape, minval=rng_pair[0], maxval=rng_pair[1])

        keys = jax.random.split(rng, 12)

        geom_friction = model.geom_friction.at[floor_id, 0].set(u(keys[0], r["floor_friction"]))

        dof_frictionloss = model.dof_frictionloss.at[dof].multiply(u(keys[1], r["frictionloss_scale"], (nj,)))
        dof_damping = model.dof_damping.at[dof].multiply(u(keys[2], r["damping_scale"], (nj,)))
        dof_armature = model.dof_armature.at[dof].multiply(u(keys[3], r["armature_scale"], (nj,)))

        # Position actuator tau = kp (target - q) - kv qdot: gain and the two bias terms must move together.
        kp = u(keys[4], r["kp_scale"], (nj,))
        kv = u(keys[5], r["kv_scale"], (nj,))
        gainprm = model.actuator_gainprm.at[:, 0].multiply(kp)
        biasprm = model.actuator_biasprm.at[:, 1].multiply(kp).at[:, 2].multiply(kv)
        forcerange = model.actuator_forcerange * u(keys[6], r["torque_limit_scale"], (nj, 1))

        body_ipos = model.body_ipos.at[torso_id].add(u(keys[7], r["torso_com_offset"], (3,)))
        body_mass = model.body_mass * u(keys[8], r["link_mass_scale"], (model.nbody,))
        body_mass = body_mass.at[torso_id].add(u(keys[9], r["torso_mass_offset"]))

        qpos0 = model.qpos0.at[consts.NUM_ROOT_QPOS:].add(u(keys[10], r["joint_zero_offset"], (nj,)))

        return (geom_friction, dof_frictionloss, dof_damping, dof_armature, gainprm, biasprm, forcerange,
                body_ipos, body_mass, qpos0)

    (geom_friction, dof_frictionloss, dof_damping, dof_armature, gainprm, biasprm, forcerange,
     body_ipos, body_mass, qpos0) = rand_dynamics(rng)

    fields = {
        "geom_friction": geom_friction,
        "dof_frictionloss": dof_frictionloss,
        "dof_damping": dof_damping,
        "dof_armature": dof_armature,
        "actuator_gainprm": gainprm,
        "actuator_biasprm": biasprm,
        "actuator_forcerange": forcerange,
        "body_ipos": body_ipos,
        "body_mass": body_mass,
        "qpos0": qpos0,
    }
    in_axes = jax.tree_util.tree_map(lambda x: None, model).tree_replace({k: 0 for k in fields})
    return model.tree_replace(fields), in_axes
