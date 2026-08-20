## Why

Qwen3.8-27B decode on P150x2 currently runs at about `4.23` tokens per second. The measured median TTFT is about `0.575` seconds.

The service submits the decode graph eagerly for every token. It also transfers complete vocabulary logits to the host. Request setup allocates fresh KV and GDN buffers.

The packaged model already contains decode-trace and batched-generation building blocks. The service does not yet use those blocks for its steady-state token loop.

## What Changes

- Add a typed Nickel profile for decode tracing, buffer bounds, and performance admission.
- Capture only the static decode graph after model warm-up.
- Snapshot and restore GDN recurrent state around trace capture.
- Keep token, position, RoPE, page-table, KV, and GDN buffers stable during each request.
- Compute greedy argmax on each vocabulary shard and transfer only bounded scalar results.
- Select the final token on the host with a deterministic global tie rule.
- Keep eager decode as a fail-closed fallback.
- Add positive and negative tests for trace bindings, state reset, shard reduction, and false performance claims.
- Retain a matched physical A/B receipt before any speedup claim.

## Impact

- **Service**: `pkgs/qwen36/service/qwen36_service` gains a traced decode path.
- **Configuration**: Nickel owns trace, buffer, timeout, and measurement bounds.
- **Runtime**: request-independent device resources remain persistent across decode steps.
- **Correctness**: eager and traced greedy continuations must match exactly.
- **Evidence**: receipts separate TTFT, decode rate, setup time, and cleanup time.

## Out of Scope

- Prefill trace capture on TT-Metal `0.74.0-rc6`.
- BF4 weight admission.
- Batch sizes above one.
- TP2 kernel-grid tuning.
- A TT-Metal pin update.
- P150x4 deployment.
- A speedup claim from simulator or device-free tests.
