import numpy as np
import jax
import jax.numpy as jnp
from jax import lax
from jax._src import config as jax_config
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas import tpu_sc as plsc
from jax.experimental.shard_map import shard_map
from jax.experimental import multihost_utils

jax_config._check_vma._set(False)


def run_sc_ring_permute(src_data, mesh, total_devices):
    """SparseCore cross-device ring permute via direct remote DMA."""
    features_per_device = src_data.shape[0] // total_devices
    hidden_dim = src_data.shape[1]
    local_shape = (features_per_device, hidden_dim)

    @pl.kernel(
        out_shape=jax.ShapeDtypeStruct(local_shape, src_data.dtype),
        mesh=plsc.ScalarSubcoreMesh(axis_name="core", num_cores=1),
        scratch_shapes=[
            pltpu.SemaphoreType.REGULAR,  # ready_sem
            pltpu.SemaphoreType.DMA,      # send_sem
            pltpu.SemaphoreType.DMA,      # recv_sem
        ],
    )
    def kernel(x_ref, y_ref, ready_sem, send_sem, recv_sem):
        my_id = lax.axis_index("x")
        axis_size = lax.axis_size("x")
        neighbor = lax.rem(my_id + 1, axis_size)

        pltpu.semaphore_signal(ready_sem, device_id=neighbor)
        pltpu.semaphore_wait(ready_sem)

        pltpu.async_remote_copy(
            x_ref, y_ref, send_sem, recv_sem, device_id=neighbor
        ).wait()

    sharding = NamedSharding(mesh, P("x", None))

    @jax.jit(in_shardings=sharding, out_shardings=sharding)
    def run(x):
        return shard_map(
            kernel,
            mesh=mesh,
            in_specs=P("x", None),
            out_specs=P("x", None),
            check_rep=False,
        )(x)

    return run(src_data)


def main():
    jax.distributed.initialize()

    total_devices = jax.device_count()
    process_id = jax.process_index()

    if process_id == 0:
        print(f"Devices: {total_devices}, Processes: {jax.process_count()}")

    devices = np.array(jax.devices())
    mesh = Mesh(devices, axis_names=("x",))

    features_per_device = 8
    hidden_dim = 128

    raw = np.zeros(
        (total_devices * features_per_device, hidden_dim), dtype=np.float32
    )
    for dev_id in range(total_devices):
        s, e = dev_id * features_per_device, (dev_id + 1) * features_per_device
        raw[s:e, :] = float(dev_id)

    sharding = NamedSharding(mesh, P("x", None))
    src = jax.device_put(raw, sharding)

    if process_id == 0:
        print("Running SC cross-device ring permute...")

    result = run_sc_ring_permute(src, mesh, total_devices)
    result.block_until_ready()

    if process_id == 0:
        print("Done. Gathering results...")

    global_result = multihost_utils.process_allgather(result, tiled=True)

    if process_id == 0:
        host = np.array(global_result)
        success = True

        for dev_id in range(total_devices):
            s = dev_id * features_per_device
            e = s + features_per_device
            sender = (dev_id - 1) % total_devices
            expected = float(sender)
            actual = host[s:e, :]

            if not np.allclose(actual, expected):
                print(
                    f"FAIL: device {dev_id} expected {expected}, "
                    f"got {actual[0, 0]}"
                )
                success = False
                break

        if success:
            print(
                f"PASS: {total_devices} devices, SC direct DMA ring permute "
                f"({features_per_device}x{hidden_dim} per device)"
            )

    jax.distributed.shutdown()


if __name__ == "__main__":
    main()
