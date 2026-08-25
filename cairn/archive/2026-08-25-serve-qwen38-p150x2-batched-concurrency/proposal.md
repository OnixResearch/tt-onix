# Serve Qwen3.8-27B P150x2 decode with batched concurrency

## Why

The served decode path is latency-bound, proven on hardware:

- served **eager ~3.7 t/s**; served **traced ~17 t/s** (host-bound removed, production
  wrapper already switched).
- The decode path is latency-bound, not throughput-bound: one GDN layer's per-step
  time is flat across batch (4-6 ms at B=1..16), so aggregate tokens/s scales almost
  linearly with active batch (layer-only: ~207 device t/s at B=1 -> ~2505 at B=16).
- The 0.74 service serves exactly one stream (one decode slot, batch-one trace), so the
  hardware's concurrency headroom is unused.

## What Changes

- A batched/concurrent decode service path: schedule multiple active streams, assemble
  each step's active batch, run one full-model decode at the active width, distribute
  per-user outputs, and keep per-user recurrent/KV state.
- A pure, device-free batch-scheduling core (which users advance in which step, per-user
  position/state slices) consumed by the service shell.
- Matched physical evidence: concurrent-served tokens/s versus the current ~17 t/s
  single-stream floor, under the existing decode admission discipline.

## Impact

- Files: batch-scheduling core, service decode adapter / server concurrency, packaged
  check probes, full-model batch measurement harness.
- Testing: positive and negative device-free scheduling tests, packaged check, full
  device-free check, and one matched concurrent-vs-single-stream physical pair.
- Non-claims: no admission claim until the matched pair exists; single-stream latency
  per user is not expected to improve (the win is aggregate).
