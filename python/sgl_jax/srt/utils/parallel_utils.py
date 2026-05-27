import logging

import jax
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from sgl_jax.global_config import global_config

logger = logging.getLogger(__name__)
_LOGGED_DECISIONS: set[tuple] = set()


def should_scatter(dim_size: int, num_devices: int) -> bool:
    """Return True if a row-parallel output should be reduce-scattered on the
    token dim. Requires per-device slice to be at least
    ``tpu_scatter_min_local_size`` and the dim to divide evenly across devices.
    """
    if num_devices <= 1:
        return False
    return (
        dim_size >= num_devices * global_config.tpu_scatter_min_local_size
        and dim_size % num_devices == 0
    )


def make_reduce_sharding(
    arr: jax.Array,
    mesh: jax.sharding.Mesh,
    *,
    scatter_dim: int = 0,
    enable_sp: bool = True,
) -> NamedSharding:
    """Output sharding for a row-parallel reduce on the 'tensor' mesh axis.

    The contracted dim consumes 'tensor' upstream; this function only
    describes where the result lands:
      - SP:  ``scatter_dim`` carries ('data', 'tensor') -> psum_scatter
      - DP:  ``scatter_dim`` carries 'data'             -> plain psum
    All other dims are replicated.

    SP triggers only when ``enable_sp`` is True and
    ``arr.shape[scatter_dim]`` clears the per-device threshold (see
    ``should_scatter``). Pass ``enable_sp=False`` to force DP regardless
    of size.
    """
    dim = arr.shape[scatter_dim]
    n_tensor = mesh.shape["tensor"]
    thr = global_config.tpu_scatter_min_local_size
    sp_active = enable_sp and should_scatter(dim, n_tensor)
    if sp_active:
        axes: str | tuple[str, ...] = ("data", "tensor")
    else:
        axes = "data"
    # One-shot log per (dim, n_tensor, thr, enable_sp, sp_active) to confirm
    # which path actually runs without spamming.
    key = (dim, n_tensor, thr, enable_sp, sp_active)
    if key not in _LOGGED_DECISIONS:
        _LOGGED_DECISIONS.add(key)
        logger.info(
            "[SP-DECISION] dim=%d n_tensor=%d thr=%d enable_sp=%s sp_active=%s " "→ axes=%s",
            dim,
            n_tensor,
            thr,
            enable_sp,
            sp_active,
            axes,
        )
    spec: list[str | tuple[str, ...] | None] = [None] * arr.ndim
    spec[scatter_dim] = axes
    return NamedSharding(mesh, P(*spec))
