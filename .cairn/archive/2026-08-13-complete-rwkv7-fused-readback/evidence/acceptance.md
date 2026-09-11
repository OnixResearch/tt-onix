# Acceptance evidence

## Result

The diagnostic-only two-program path passed exact physical parity on one Blackhole device. The HiFi4 program produced the recurrent state. The separate HiFi2 program produced the raw BF16 readout. The caller reshaped the readout and applied the attention correction exactly once.

The physical receipt has BLAKE3 `f53b9364e95295188ce1c48eec27014e8fa90d37c1c0bc813b83d578eaca7264`. The repository retains its aggregate facts in `physical-diagnostic-result.json`. Raw runtime, compiler, watcher, cache, and profiler files remain outside the repository.

## Exact gates

- Full split recurrent-state parity passed with maximum absolute error `0.0`.
- Raw BF16 readout parity passed with maximum absolute error `0.0`.
- Corrected-row BF16 parity passed with maximum absolute error `0.0`.
- The split result reported `correction_complete = false`.
- The caller owned and applied correction exactly once.
- No generic device operation ran before the diagnostic launches.
- The generic control ran after synchronized diagnostic readback.
- The receipt was written after the bounded terminal stage.

## JIT and safety evidence

Generated JIT descriptors bind recurrence to HiFi4 and readout to HiFi2. Both use exact approximation mode. The recurrence descriptors contain the FP32 recurrence and split-readout defines. Reader include descriptors bind the immutable diagnostic runtime sources.

The package-matched Blackhole LLK audit covered all nine hazard classes and the cross-class JOIN pass. It found no live or emergent race. See `llk-hazard-audit.md`.

## Claim boundary

This evidence accepts the diagnostic correction only. It makes no performance or production admission claim. It does not reuse the rejected window-16 candidate. Production remains profile `rwkv7-p150x2-persistent-decode-v3` with windows `[2, 4, 8]`.

A later change must create a new immutable window-16 candidate. That candidate must pass every simulator, physical correctness, performance, canary, monitoring, deployment, and observation gate before admission.
