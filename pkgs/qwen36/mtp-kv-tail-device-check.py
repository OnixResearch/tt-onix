"""Bounded, model-free device check for exact rejected target-KV restoration."""

import argparse
import json
import os
from contextlib import ExitStack
from types import SimpleNamespace

import torch
import ttnn
from qwen36_service.mtp_kv_tail_adapter import MtpTargetKvRecovery
from qwen36_service.mtp_state_snapshot_adapter import MtpStateSnapshot, tensor_identity

PAGE_COUNT = 4
HEAD_COUNT = 2
PAGE_SIZE = 128
CHANNELS = 256
DRAFT_COUNT = 2
PHYSICAL_COLUMNS = 2
PAGES = (2, 0, 3, 1)
SENTINEL_MODULUS = 97
SENTINEL_OFFSET = 48
WRITE_VALUE = 200
CACHE_SHAPE = (PAGE_COUNT, HEAD_COUNT, PAGE_SIZE, CHANNELS)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--columns", type=int, choices=(1, PHYSICAL_COLUMNS), default=1)
    args = parser.parse_args()
    comparisons = 0
    with ExitStack() as cleanup:
        mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, args.columns))
        cleanup.callback(ttnn.close_mesh_device, mesh)
        shard = ttnn.create_sharded_memory_config(
            shape=(ttnn.TILE_SIZE, CHANNELS),
            core_grid=ttnn.CoreGrid(x=1, y=1),
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

        def to_device(host):
            return ttnn.from_torch(
                host,
                device=mesh,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
            )

        element_count = PAGE_COUNT * HEAD_COUNT * PAGE_SIZE * CHANNELS
        host = (
            torch.arange(element_count).reshape(CACHE_SHAPE) % SENTINEL_MODULUS
            - SENTINEL_OFFSET
        ).to(torch.bfloat16)
        # Preserve signed zero as bits, not only as numerically equal values.
        host[0, 0, 1, 0] = -0.0
        entry = (host, -host)
        allocated_caches = []
        for value in entry:
            cache = to_device(value)
            cleanup.callback(ttnn.deallocate, cache)
            allocated_caches.append(cache)
        caches = tuple(allocated_caches)
        bindings = tuple(tensor_identity(cache) for cache in caches)
        snapshot = MtpStateSnapshot(lambda: caches)
        cleanup.callback(snapshot.close)
        attention = SimpleNamespace(
            paged_k=caches[0], paged_v=caches[1], _kv_shard_cfg=lambda _: shard
        )
        model = SimpleNamespace(
            layers=(SimpleNamespace(is_full_attention=True, attention=attention),)
        )
        position = PAGE_SIZE - 1
        recovery = MtpTargetKvRecovery(
            model=model,
            snapshot=snapshot,
            mesh=mesh,
            page_table=torch.tensor([PAGES], dtype=torch.int32),
            position=position,
            draft_count=DRAFT_COUNT,
        )
        cleanup.callback(recovery.close)
        expected = [value.clone() for value in entry]
        acceptance = (0, 1, DRAFT_COUNT, 0)
        for accepted in acceptance:
            for cache, current in zip(caches, expected, strict=True):
                for index in range(position, position + DRAFT_COUNT + 1):
                    current[PAGES[index // PAGE_SIZE], :, index % PAGE_SIZE, :] = (
                        WRITE_VALUE + index % HEAD_COUNT
                    )
                source = to_device(current)
                try:
                    ttnn.copy(source, cache)
                finally:
                    ttnn.deallocate(source)
            for index in range(position + accepted + 1, position + DRAFT_COUNT + 1):
                page, row = PAGES[index // PAGE_SIZE], index % PAGE_SIZE
                for current, original in zip(expected, entry, strict=True):
                    current[page, :, row, :] = original[page, :, row, :]
            recovery.finish_round(position=position, accepted=accepted)
            for cache, current in zip(caches, expected, strict=True):
                devices = ttnn.get_device_tensors(cache)
                assert len(devices) == args.columns, "device count changed"
                for device_tensor in devices:
                    actual = ttnn.to_torch(device_tensor)
                    actual_bits = actual.contiguous().view(torch.int16)
                    expected_bits = current.contiguous().view(torch.int16)
                    if not torch.equal(actual_bits, expected_bits):
                        different = actual_bits != expected_bits
                        first = tuple(torch.nonzero(different)[0].tolist())
                        print(
                            json.dumps({
                                "stage": "cache-mismatch",
                                "position": position,
                                "accepted": accepted,
                                "count": int(different.sum()),
                                "first_index": first,
                                "actual": float(actual[first]),
                                "expected": float(current[first]),
                                "actual_bits": int(actual_bits[first]),
                                "expected_bits": int(expected_bits[first]),
                            }),
                            flush=True,
                        )
                        raise AssertionError(
                            "cache bits changed outside the required recovery"
                        )
                    comparisons += 1
            assert tuple(tensor_identity(cache) for cache in caches) == bindings, (
                "cache allocations changed"
            )
            position += accepted + 1
        try:
            recovery.finish_round(position=position + 1, accepted=0)
        except ValueError:
            pass
        else:
            raise AssertionError("skipped boundary was accepted")
        recovery.close()
        try:
            recovery.finish_round(position=position, accepted=0)
        except RuntimeError:
            pass
        else:
            raise AssertionError("closed recovery owner was accepted")
    print(
        json.dumps({
            "passed": True,
            "simulated": bool(os.environ.get("TT_METAL_SIMULATOR")),
            "mesh_columns": args.columns,
            "rounds": len(acceptance),
            "bitwise_cache_comparisons": comparisons,
            "negative_cases": ["skipped-boundary", "closed-owner"],
            "cleanup_complete": True,
            "admission_eligible": False,
        })
    )


if __name__ == "__main__":
    main()
