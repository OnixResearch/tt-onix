# r[impl tenstorrent.training_platform.launch]
# Bounded collective-free smoke operation for the p150_x2 (1x2) mesh.
# Runs inside the pinned TT-XLA runtime image. Prints one JSON result line.
import json
import sys

EXPECTED_DEVICE_COUNT = 2


def main() -> int:
    import numpy as np

    import jax

    devices = jax.devices("tt")
    if len(devices) != EXPECTED_DEVICE_COUNT:
        print(
            json.dumps(
                {
                    "status": "fail",
                    "stage": "mesh_open",
                    "detail": f"expected {EXPECTED_DEVICE_COUNT} TT devices, saw {len(devices)}",
                }
            )
        )
        return 1

    mesh = jax.sharding.Mesh(np.array(devices).reshape(1, EXPECTED_DEVICE_COUNT), ("x",))
    sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("x", None))

    source = np.arange(8, dtype=np.float32).reshape(2, 4)
    placed = jax.device_put(source, sharding)
    computed = placed + 1.0
    observed = np.asarray(computed)
    expected = source + 1.0

    if not np.array_equal(observed, expected):
        print(
            json.dumps(
                {
                    "status": "fail",
                    "stage": "smoke_execute",
                    "detail": "collective-free tensor operation returned wrong values",
                }
            )
        )
        return 1

    print(
        json.dumps(
            {
                "status": "pass",
                "mesh": [1, EXPECTED_DEVICE_COUNT],
                "device_count": len(devices),
                "values_exact": True,
            }
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
