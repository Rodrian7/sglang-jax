import jax
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from sgl_jax.global_config import global_config


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
    if enable_sp and should_scatter(arr.shape[scatter_dim], mesh.shape["tensor"]):
        # VERIFY-ONLY SENTINEL: catch Bug 1 (divisibility) and Bug 2 (threshold
        # device count). Production code passes only `tensor` to should_scatter,
        # but the actual scatter runs across `data * tensor`. If either invariant
        # fails on `world`, the SP path was wrongly approved.
        dim = arr.shape[scatter_dim]
        world = mesh.shape["data"] * mesh.shape["tensor"]
        threshold = global_config.tpu_scatter_min_local_size
        assert dim % world == 0, (
            f"[BUG1] should_scatter approved dim={dim} but dim % world={world} != 0 "
            f"(tensor={mesh.shape['tensor']}, data={mesh.shape['data']}); "
            f"JAX reshard on ('data','tensor') will fail."
        )
        per_device = dim // world
        assert per_device >= threshold, (
            f"[BUG2] should_scatter approved SP with per-device={per_device} "
            f"< threshold={threshold} (dim={dim}, world={world}, "
            f"tensor={mesh.shape['tensor']}, data={mesh.shape['data']}). "
            f"should_scatter used tensor not world as num_devices."
        )
        axes: str | tuple[str, ...] = ("data", "tensor")
    else:
        axes = "data"
    spec: list[str | tuple[str, ...] | None] = [None] * arr.ndim
    spec[scatter_dim] = axes
    return NamedSharding(mesh, P(*spec))
