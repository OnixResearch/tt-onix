"""Compare two Qwen readback adapters on physical TT-NN tensors, not a full model."""

import argparse
import importlib.util
import json
import os
import statistics
import sys
import time
from pathlib import Path

import torch
import ttnn

DEVICE_COUNT = 2
WIDTH = 4
VOCABULARY_PER_SHARD = 16
QUEUE_ID = 0
COMMAND_QUEUE_COUNT = 2
DEFAULT_ITERATIONS = 1000
WARMUP_ITERATIONS = 100
EXECUTION_ORDER = ("baseline", "candidate", "candidate", "baseline")


def load_adapter(name, path, mesh):
    spec = importlib.util.spec_from_file_location(f"qwen36_service.{name}", path)
    if spec is None or spec.loader is None:
        raise ValueError("adapter module cannot be loaded")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    adapter = object.__new__(module.BatchedTracedDecodeAdapter)
    adapter.mesh = mesh
    adapter.device_count = DEVICE_COUNT
    adapter.vocabulary_per_shard = VOCABULARY_PER_SHARD
    adapter.vocabulary_size = VOCABULARY_PER_SHARD * DEVICE_COUNT
    adapter._mesh_reader = ttnn.ConcatMeshToTensor(mesh, dim=0)
    return adapter._read_global_winners


def measure(accessor, indices, values, iterations, expected):
    started = time.perf_counter_ns()
    for _ in range(iterations):
        result = accessor(indices, values, WIDTH)
        if result != expected:
            raise AssertionError("readback changed or returned stale winners")
    return (time.perf_counter_ns() - started) / iterations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS)
    args = parser.parse_args()
    if args.iterations <= 0:
        parser.error("iterations must be positive")
    mesh = ttnn.open_mesh_device(
        mesh_shape=ttnn.MeshShape(1, DEVICE_COUNT),
        physical_device_ids=list(range(DEVICE_COUNT)),
        num_command_queues=COMMAND_QUEUE_COUNT,
    )
    try:
        adapters = {
            name: load_adapter(f"readback_{name}", path, mesh)
            for name, path in (
                ("baseline", args.baseline),
                ("candidate", args.candidate),
            )
        }
        mapper = ttnn.ShardTensorToMesh(mesh, dim=0)
        shape = (DEVICE_COUNT, 1, 1, WIDTH)
        indices = ttnn.from_torch(
            torch.arange(WIDTH).repeat(DEVICE_COUNT, 1).reshape(shape),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh,
            mesh_mapper=mapper,
        )
        source_values = torch.tensor(((1, 0, 1, 0), (0, 1, 1, 0)), dtype=torch.bfloat16)
        values = ttnn.from_torch(
            source_values.reshape(shape),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh,
            mesh_mapper=mapper,
        )
        ttnn.synchronize_device(mesh)
        # Reuse the same device addresses while changing the winning shard.
        for variant in (0, 1, 0):
            source = source_values if variant == 0 else source_values.flip((0,))
            host = ttnn.from_torch(
                source.reshape(shape),
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=mapper,
            )
            ttnn.copy_host_to_device_tensor(host, values, cq_id=QUEUE_ID)
            expected = tuple(
                VOCABULARY_PER_SHARD + column if column == 1 - variant else column
                for column in range(WIDTH)
            )
            for accessor in adapters.values():
                if accessor(indices, values, WIDTH) != expected:
                    raise AssertionError(
                        "readback missed a buffer update or changed ties"
                    )
        # A failed second read must drain the pending first read and permit retry.
        try:
            adapters["candidate"](indices, None, WIDTH)
        except (AttributeError, RuntimeError, TypeError):
            pass
        else:
            raise AssertionError("invalid second tensor was accepted")
        if adapters["candidate"](indices, values, WIDTH) != expected:
            raise AssertionError("readback did not recover after a rejected transfer")
        invalid_indices = ttnn.from_torch(
            torch.full(shape, VOCABULARY_PER_SHARD * DEVICE_COUNT),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh,
            mesh_mapper=mapper,
        )
        for accessor in adapters.values():
            try:
                accessor(invalid_indices, values, WIDTH)
            except RuntimeError as error:
                if "outside the vocabulary" not in str(error):
                    raise
            else:
                raise AssertionError("out-of-vocabulary winner was accepted")
            measure(accessor, indices, values, WARMUP_ITERATIONS, expected)
        samples = {name: [] for name in adapters}
        for name in EXECUTION_ORDER:
            samples[name].append(
                measure(adapters[name], indices, values, args.iterations, expected)
            )
        baseline_ns = statistics.median(samples["baseline"])
        candidate_ns = statistics.median(samples["candidate"])
        result = {
            "schema_version": 1,
            "simulated": bool(os.environ.get("TT_METAL_SIMULATOR")),
            "requested_device_ids": list(range(DEVICE_COUNT)),
            "command_queue_count": COMMAND_QUEUE_COUNT,
            "read_queue_id": QUEUE_ID,
            "baseline_source": str(args.baseline),
            "candidate_source": str(args.candidate),
            "iterations_per_sample": args.iterations,
            "execution_order": EXECUTION_ORDER,
            "nanoseconds_per_call": samples,
            "speedup_ratio": baseline_ns / candidate_ns,
            "exact_winner_parity": True,
            "buffer_updates_and_ties_checked": True,
            "failed_read_retry_passed": True,
            "invalid_winner_rejected": True,
            "non_claims": ["not a full-model or end-to-end decode benchmark"],
        }
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
