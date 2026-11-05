"""Compatibility shims for the installed library versions. Import before brax is used; delete when brax catches up.

brax 0.14.2 (the latest release) still calls jax.device_put_replicated, which jax 0.11 removed.
"""
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec


def _device_put_replicated(tree, devices):
    """Same result as the removed jax.device_put_replicated: a leading axis of len(devices), one slice per device."""
    sharding = NamedSharding(Mesh(devices, ("d",)), PartitionSpec("d"))
    return jax.tree.map(lambda x: jax.device_put(jnp.stack([x] * len(devices)), sharding), tree)


try:
    jax.device_put_replicated        # still there in older jax: leave it alone
except AttributeError:
    jax.device_put_replicated = _device_put_replicated
