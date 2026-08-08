# r[impl tenstorrent.training_platform.launch]
# Bounded collective-free smoke operation for the p150_x2 (1x2) mesh.
# Runs inside the pinned TT-XLA runtime image. Prints one JSON result line.
import json
import sys

EXPECTED_DEVICE_COUNT = 2


def emit(status: str, stage: str, **extra) -> None:
    payload = {"status": status, "stage": stage}
    payload.update(extra)
    print(json.dumps(payload))


def main(current_stage: list) -> int:
    import jax
    import numpy as np

    # The upstream multichip example disables the shardy partitioner; the
    # default partitioner path fails on this PJRT plugin version.
    jax.config.update("jax_use_shardy_partitioner", False)

    devices = jax.devices("tt")
    if len(devices) != EXPECTED_DEVICE_COUNT:
        emit(
            "fail",
            "mesh_open",
            detail=f"expected {EXPECTED_DEVICE_COUNT} TT devices, saw {len(devices)}",
        )
        return 1
    emit("pass", "mesh_open", device_count=len(devices))
    current_stage[0] = "smoke_execute"

    # Mirror the upstream multichip example: shard_map inside jit, inputs
    # moved to the mesh with device_put. Plain jit with presharded arguments
    # fails the StableHLO pipeline on this PJRT plugin version.
    from jax.experimental.shard_map import shard_map
    from jax.sharding import NamedSharding
    from jax.sharding import PartitionSpec as P

    mesh = jax.make_mesh((len(devices),), ("data",), devices=devices)
    inputs_specs = P("data")

    source = np.arange(8, dtype=np.float32).reshape(2, 4)
    device_inputs = jax.device_put(source, NamedSharding(mesh, inputs_specs))
    mapped = jax.jit(
        shard_map(
            lambda value: value + 1.0,
            mesh,
            in_specs=inputs_specs,
            out_specs=inputs_specs,
            check_rep=False,
        ),
        out_shardings=NamedSharding(mesh, inputs_specs),
    )
    computed = mapped(device_inputs)
    observed = np.asarray(jax.device_get(computed))
    expected = source + 1.0

    if not np.array_equal(observed, expected):
        emit(
            "fail",
            "smoke_execute",
            detail="collective-free tensor operation returned wrong values",
            observed=observed.tolist(),
            expected=expected.tolist(),
            observed_shape=list(observed.shape),
            observed_dtype=str(observed.dtype),
        )
        return 1

    emit("pass", "smoke_execute", mesh=[1, EXPECTED_DEVICE_COUNT], values_exact=True)
    return 0


if __name__ == "__main__":
    current_stage = ["mesh_open"]
    try:
        sys.exit(main(current_stage))
    # This process boundary converts every framework error into a terminal receipt.
    except Exception as error:  # noqa: BLE001
        emit("fail", current_stage[0], detail=f"{type(error).__name__}: {error}")
        sys.exit(1)
