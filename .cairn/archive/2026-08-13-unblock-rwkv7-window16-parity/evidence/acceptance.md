# Window-16 fused-only diagnostic result

## Result

Window `16` remains blocked. Production still admits only `[2, 4, 8]`.

The diagnostic package now reaches the three fused kernels without generic TT-NN device pre-work. The executable and kernel root resolve to `rwkv7-p150x2-window16-readout-diagnostic-runtime`. The BF16 readout setting is enabled only in that diagnostic package.

## Corrected timeout diagnosis

The earlier archived probe compiled 47 generic TT-NN kernels before the three fused kernels. Its retained launcher also resolved to the rejected candidate runtime.

The new probe uses a pure host reference. Its watcher list contains only the three fused user kernels, plus dispatch kernels. This fixes the original compile-path and package-binding defects.

## Physical result

The first bounded attempt used device `0` and a 30-minute timeout. All three fused kernels compiled within roughly 34 seconds after device open. The program then reached `fused-launch-finished`, but synchronized tensor readback did not finish before the timeout. No correctness receipt was written.

The device-open log also recorded repeated failures from the current always-on real-time profiler. The requested profiler-disable flags did not disable that service. This was a confirmed harness defect.

The package-matched Blackhole descriptor and TT-Metal grid parser exclude the relative last-column dispatch cores from the fast-dispatch compute grid. The fused worker set uses only the first 16 logical compute cores. Therefore, it does not overlap the fast-dispatch service cores.

The authorized harness retry used Ethernet dispatch to make the real-time profiler ineligible. It did not change the runtime or fused kernels. Device firmware initialization failed before `device-opened`, after a 10-second wait for physical cores. It wrote no receipt.

Both Tenstorrent devices were unowned after each attempt.

Aggregate evidence is in `physical-diagnostic-result.json`. Raw logs and generated kernel paths remain outside the repository.

## Decision

Exact FP32 recurrent-state parity and exact BF16 corrected-row parity were not evaluated. Therefore, this change does not confirm the BF16 correction.

Do not create a new immutable window-16 candidate. Do not start simulator, full-model correctness, profiling, A/B, canary, monitoring, deployment, or observation gates.

A later change must first make fused-program completion and readback finish within a named bound. It must then write the exact asymmetric parity receipt.
