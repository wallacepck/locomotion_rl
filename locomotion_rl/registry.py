"""Registry for the custom robot's environments (counterpart of mujoco_playground.registry for the Go1 ones).

Playground's registry only knows its own robots, and its `load` additionally clones the Menagerie repository, which
we do not use. So this is a small, separate table with the same four lookups the training scripts need:

    default config  ->  environment  ->  domain randomizer  ->  PPO config

    from locomotion_rl import registry
    env = registry.load("CustomRobotJoystickFlatTerrain", num_envs=4096)
"""
import dataclasses
import functools
from typing import Any, Callable, Dict, Optional

from ml_collections import config_dict

from locomotion_rl.custom_robot import joystick, params, randomize


@dataclasses.dataclass(frozen=True)
class Entry:
    env_class: Callable[..., Any]
    default_config: Callable[[], config_dict.ConfigDict]
    randomizer: Callable
    ppo_config: Callable[[str], config_dict.ConfigDict]


_ENTRIES: Dict[str, Entry] = {
    "CustomRobotJoystickFlatTerrain": Entry(
        functools.partial(joystick.Joystick, task="flat_terrain"), joystick.default_config,
        randomize.domain_randomize, params.brax_ppo_config
    ),
    "CustomRobotJoystickRoughTerrain": Entry(
        functools.partial(joystick.Joystick, task="rough_terrain"), joystick.default_config,
        randomize.domain_randomize, params.brax_ppo_config
    ),
    "CustomRobotJoystickBlockTerrain": Entry(
        functools.partial(joystick.Joystick, task="block_terrain"), joystick.default_config,
        randomize.domain_randomize, params.brax_ppo_config
    ),
}

ALL_ENVS = tuple(_ENTRIES)


def _entry(env_name: str) -> Entry:
    if env_name not in _ENTRIES:
        raise ValueError(f"Env '{env_name}' not found. Available envs: {ALL_ENVS}")
    return _ENTRIES[env_name]


def get_default_config(env_name: str) -> config_dict.ConfigDict:
    return _entry(env_name).default_config()


def get_domain_randomizer(env_name: str) -> Callable:
    return _entry(env_name).randomizer


def get_ppo_config(env_name: str) -> config_dict.ConfigDict:
    return _entry(env_name).ppo_config(env_name)


def load(
    env_name: str,
    config: Optional[config_dict.ConfigDict] = None,
    config_overrides: Optional[Dict[str, Any]] = None,
    num_envs: Optional[int] = None,
):
    """Environment instance. `num_envs` sizes the warp contact buffer (config.naconmax) for that many parallel worlds."""
    entry = _entry(env_name)
    config = (config or entry.default_config()).copy_and_resolve_references()   # MjxEnv applies overrides in place
    if num_envs is not None and config.impl == "warp":
        config_overrides = {"naconmax": params.NACON_PER_ENV * num_envs, "naccdmax": params.CCD_PER_ENV * num_envs,
                            **(config_overrides or {})}
    return entry.env_class(config=config, config_overrides=config_overrides)
