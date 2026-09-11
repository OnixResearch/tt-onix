# Qwen binding-identity optimization

## Result

The binding accessor is about twice as fast in a bounded microbenchmark on physical P150x2 hardware. Median lookup time fell from approximately 1.02 microseconds to 0.51 microseconds.

This is a small host-side optimization. It is not a measured doubling of Qwen decode speed. No full-request speedup is claimed.

A subsequent full-model smoke passed. One B=4 cohort generated 64 tokens per request and matched the earlier reference outputs exactly. The production service remains stopped.

## Change and contract

`pkgs/qwen36/service/qwen36_service/batch_decode_adapter.py` still validates every binding on every token. It does not cache identities or remove synchronization.

The old accessor constructs temporary tensor shard views and reads each address. The new path reads the shared address once and repeats it for the coordinate count. It preserves the exact identity string, including the shard count.

The shortcut applies only to the exact native `ttnn.Tensor` type. Subclasses, host tensors, unallocated tensors, and unavailable APIs retain the legacy fallback. Buffer replacement, deallocation, and shard-count changes remain detectable.

The package-matched TT-Metal 0.77 source establishes the equivalence:

- `ttnn/core/distributed/api.cpp`, `get_device_tensors`: device shards are views of the same `DeviceStorage`.
- `ttnn/cpp/ttnn-nanobind/pytensor.cpp`: `buffer_address` returns `self.mesh_buffer().address()`.
- The same file exposes the storage coordinate set through `device_coords`.
- `tt_metal/distributed/mesh_buffer.cpp` creates device buffers at the shared `address_`.

The `buffer_address` docstring still says single-device only. The 0.77 implementation and the physical two-device test support the shared-address behavior. This source/doc mismatch is explicit, not a claim about every TT-NN release.

The source pin is the one recorded in [the 0.77 port report](tt-metal-0.77-port.md).

## Measurements and tests

The [accessor receipt](evidence/tt-metal-0.77/qwen-binding-identity/accessor.json) records 10,000 calls per sample in baseline, candidate, candidate, baseline order. It also records exact identity parity, replacement detection, and deallocation detection.

The [model receipt](evidence/tt-metal-0.77/qwen-binding-identity/model-parity.json) compares four saved responses with the earlier host-limited reference. A deliberately corrupted token fails the comparison. This is a correctness smoke, not a new performance-admission run.

Validation passed:

- Existing Qwen checks and basic simulator smoke before the change.
- Qwen package and opt-in host-wrapper checks after the change.
- New adapter fixtures for direct lookup, address and shard-count changes, empty coordinates, subclass fallback, unsupported APIs, and host fallback.
- The real-tensor accessor probe, including released and host tensors.
- The model smoke, its token-corruption control, and rejection of an empty response corpus.
- Ruff for the new probe and rejection of a zero iteration count before device access.

Negative accessor tests intentionally trigger native diagnostics for host or released buffers. Both hardware probes closed the drivers normally.

The existing adapter retains its pre-existing import-order lint finding. Repository-wide lint success is not claimed.

## Reproduction and limits

The diagnostic source is `pkgs/qwen36/binding-identity-probe.py`. Its default mode compares two accessor modules on a 1x2 mesh. Its `model-responses` mode compares saved response files without opening a device.

The baseline package is `/nix/store/qw9llks4k65np226xygfrmzx1qcbs6nz-qwen36-0.1.0`. The tested candidate is `/nix/store/acqd15fjnx3gcilph1vc80q7qkqw2m9h-qwen36-0.1.0`.

The hardware budget was one 300-second accessor probe and one 900-second model probe. Both used private scratch space. The model probe retained read-only weights and caches, the 96 GiB memory limit, and explicit host thread limits.

Queue synchronization was a separate approach. The service has two command queues, so no full-mesh barrier was removed without a separate ordering proof. The validated change only avoids temporary shard construction.

The change updates the execution identity. Earlier full-request timing results remain attached to their original package. Broad model accuracy, sustained decode acceleration, and production admission remain outside this result. No deployment, firmware change, or push occurred.
