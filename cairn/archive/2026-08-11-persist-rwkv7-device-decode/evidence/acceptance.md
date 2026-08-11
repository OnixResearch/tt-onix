# Persistent decode acceptance

## Verdict

Production admission passed on 2026-08-11.

The installed profile selects bounded Metal Trace for eligible workloads. All other workloads use normal dispatch.

## Bounds

- The trace has two decode steps.
- The output capacity is two tokens per request.
- The queue depth is four requests per card.
- The elapsed-time bound is 60 seconds.
- The trace region is 256 MiB.
- One host replay submission executes both decode steps.

## Correctness

The admitted workload uses prompt tokens `[0, 2]`. Its pinned FP32 CPU-reference output is `[2, 2]`.

The production receipt has exact token parity and exact FP32 recurrent-state parity. The final state BLAKE3 equals the normal-dispatch state BLAKE3.

All four lifecycle receipts are complete and releasable. The runtime acknowledges completion and zeros state before it releases each slot.

The retained production receipt is `production-profiled-worker-receipt.json`.

## Timing boundary

Normal A/B reference runs set `RWKV7_PERSISTENT_DECODE_CANDIDATE=0`. This control keeps normal dispatch and moves the shared prompt prefix before the timed region.

Persistent production runs omit the variable. Trace capture, normal reference execution, readback, hashing, acknowledgement, and zeroing remain outside the timed region.

Both modes time the same final prompt token and two generated-token steps.

## Physical A/B

Five order-balanced one-card pairs passed. The median paired speedup was `2.6697484573326276x`. The minimum paired speedup was `2.59883139510804x`.

Five order-balanced P150x2 pairs passed. The four requests were split across physical devices 0 and 1. The median paired speedup was `3.1922345786773345x`. The minimum paired speedup was `3.133407266069899x`.

Every persistent receipt reports one timed host model submission. Every normal receipt reports two submissions.

The aggregate values and all raw receipt BLAKE3 identities are in `physical-acceptance.json`.

## Profiler

The final production run used these settings:

```sh
TT_METAL_DEVICE_PROFILER=1
TT_METAL_TRACE_PROFILER=1
```

The production receipt passed `rwkv7-p150x2-persistent-decode-evidence production`.

The trace-only CSV had 520 `TRACE-FW` or `TRACE-KERNEL` rows. No profiler-buffer-drop warning was present.

The raw CSV BLAKE3 is `2eba9af8f1f798fda4bf7d49c2922edb53ec46fdc56aab8baa3cbfa06aafada3`.

## Simulator

The final Blackhole `1x1` simulator smoke passed with two recurrent steps and one trace replay submission.

The simulator receipt states `physical_hardware = false`, `performance_claim = false`, and `simulator_determinism_claim = false`.

The retained receipt is `simulator-receipt.json`.

## Fail-closed checks

A one-step workload was ineligible. Production selected `normal_dispatch` and reported `trace_unavailable` without entering the trace.

The pure lifecycle tests reject stale bindings, duplicate slots, invalid transitions, output overflow, missing cancellation, timeout, parity loss, and incomplete cleanup.

Static searches found no direct semaphore, mailbox, circular-buffer, NoC, MMIO, source-register, or raw SFPI primitive in the persistent host path. The path uses the reviewed TT-NN operations and Metal Trace API.

## Commands

The final package and simulator checks used:

```sh
nix build .#rwkv7-p150x2-runtime
nix build .#rwkv7-p150x2-evidence
rwkv7-p150x2-persistent-decode-simulator-smoke /absolute/new/output
rwkv7-p150x2-persistent-decode-evidence production /absolute/receipt.json
```

The remote builder was unavailable. Nix completed all builds locally.
