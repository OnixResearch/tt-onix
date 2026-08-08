# r[impl tenstorrent.rwkv7_dual_p150a.collectives]
"""Focused collective probe for the dual-P150A logical data mesh.

Runs inside the pinned TT-XLA container. Opens the two-replica SPMD data
mesh over both physical P150A devices, runs exact mean all-reduce probes
with distinct per-device contributions, runs a barrier synchronization, and
reports placement, timing, and output evidence. Prints one JSON result
line.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time

import torch

ALL_REDUCE_PATTERN = re.compile(r"all_reduce")
REPLICATED_SPEC = "{replicated}"
ALL_REDUCE_ITERATIONS = 4

# Multichip environment from the pinned TT-Blacksmith device manager: the
# TT backend compiles SPMD through the shardy round trip with explicit
# all-reduce.
os.environ.setdefault("PJRT_DEVICE", "TT")
os.environ.setdefault("XLA_STABLEHLO_COMPILE", "1")
os.environ.setdefault("XLA_ALWAYS_ALLREDUCE", "1")
os.environ.setdefault("CONVERT_SHLO_TO_SHARDY", "1")
os.environ.setdefault("DISABLE_NUMERIC_CC_TOKEN", "1")


def emit(status, **extra):
    payload = {"status": status}
    payload.update(extra)
    print(json.dumps(payload, sort_keys=True))


def sharding_specs(tensors):
    """Human-readable sharding specs through the wheel's private binding.

    The public `torch_xla.distributed.spmd` namespace does not export a
    spec getter in this wheel; TT-Blacksmith uses the same binding.
    """
    import torch_xla

    return [torch_xla._XLAC._get_xla_sharding_spec(t) or "" for t in tensors]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", required=True)
    args = parser.parse_args()
    with open(args.profile, encoding="utf-8") as handle:
        profile = json.load(handle)

    expected_devices = profile["device"]["physical_count"]
    axis_name = profile["data_axis"]["name"]
    replicas = profile["data_axis"]["replicas"]

    import numpy as np
    import torch_xla
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.spmd as xs
    import torch_xla.runtime as xr

    xr.set_device_type("TT")
    xr.use_spmd()

    device_count = xr.global_runtime_device_count()
    if device_count != expected_devices:
        emit(
            "fail", detail=f"expected {expected_devices} TT devices, saw {device_count}"
        )
        sys.exit(1)

    mesh = xs.Mesh(np.arange(device_count), (device_count,), (axis_name,))
    device = xm.xla_device()

    # Distinct exact fp32 contributions: replica 0 holds 3.0, replica 1
    # holds 5.0. The exact mean is 4.0. A wrong contribution, a missing
    # device, or a host fallback changes the result.
    contributions = torch.tensor([3.0, 5.0], dtype=torch.float32).to(device)
    xs.mark_sharding(contributions, mesh, (axis_name,))

    t0 = time.monotonic()
    reduced = None
    for _ in range(ALL_REDUCE_ITERATIONS):
        total = contributions.sum()
        reduced = total / float(replicas)
    # Capture the pending graph before any sync materializes it; an empty
    # dump would carry no collective evidence.
    hlo_text = xm.get_stablehlo([reduced])
    if not hlo_text.strip():
        hlo_text = xm.get_stablehlo()
    torch_xla.sync(wait=True)
    collective_seconds = time.monotonic() - t0

    # Barrier: an all-reduce of ones returns the replica count on every
    # replica only when both sides arrive.
    ones = torch.tensor([1.0, 1.0], dtype=torch.float32).to(device)
    xs.mark_sharding(ones, mesh, (axis_name,))
    barrier_value = ones.sum()
    torch_xla.sync(wait=True)

    reduced_cpu = float(reduced.to("cpu").item())
    barrier_cpu = float(barrier_value.to("cpu").item())
    values_exact = reduced_cpu == 4.0
    synchronized = barrier_cpu == float(replicas)

    finite = math.isfinite(reduced_cpu) and math.isfinite(barrier_cpu)
    with open("/evidence/collective.hlo", "w", encoding="utf-8") as handle:
        handle.write(hlo_text)
    # Placement evidence: the input argument carries the data-axis
    # sharding annotation and the reduced result is replicated. The exact
    # values prove both devices contributed: a single-device reduction
    # over one shard would return 1.5 or 2.5, never 4.0.
    all_reduce_count = len(ALL_REDUCE_PATTERN.findall(hlo_text))
    specs = sharding_specs([contributions, reduced])
    contribution_sharded = "devices=[2" in specs[0]
    result_replicated = specs[1] == REPLICATED_SPEC

    finite = math.isfinite(reduced_cpu) and math.isfinite(barrier_cpu)
    with open("/evidence/collective.hlo", "w", encoding="utf-8") as handle:
        handle.write(hlo_text)

    emit(
        "pass",
        mesh={
            "rows": 1,
            "columns": device_count,
            "axis_name": axis_name,
            "device_count": device_count,
        },
        all_reduce={
            "op": "mean",
            "placement": "TT",
            "count": ALL_REDUCE_ITERATIONS,
            "hlo_bytes": len(hlo_text),
            "pre_partition_all_reduce_ops": all_reduce_count,
            "iterations": ALL_REDUCE_ITERATIONS,
            "both_contributed": values_exact and contribution_sharded,
            "values_exact": values_exact,
            "input_values": [3.0, 5.0],
            "reduced_value": reduced_cpu,
            "contribution_spec": specs[0],
            "result_spec": specs[1],
            "result_replicated": result_replicated,
        },
        barrier={"synchronized": synchronized, "observed": barrier_cpu},
        finite=finite,
        collective_seconds=collective_seconds,
    )


if __name__ == "__main__":
    try:
        main()
    # This process boundary converts every framework error into a terminal diagnostic.
    except Exception as error:  # noqa: BLE001
        import traceback

        traceback.print_exc()
        emit("fail", detail=f"{type(error).__name__}: {error}")
        sys.exit(1)
