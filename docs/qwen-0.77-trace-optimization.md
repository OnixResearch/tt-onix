# Qwen 0.77 trace-reuse optimization

## Result

Same-width trace reuse passed the existing B=4 performance gate with explicit host thread limits. Median request time fell from 7.75 seconds to 7.20 seconds, about 7%.

Token and text outputs matched their recapture counterparts for every prompt and repetition. The test retained the original numerical and performance bounds.

The new package `qwen36-trace-reuse-host1` exposes this route as an opt-in diagnostic command. Normal service defaults remain unchanged. The production service remains stopped.

## Measured scope

The test used Qwen3.8-27B, TT-Metal 0.77.0, and physical PCIe devices 0 and 1. Each cohort contained four concurrent requests. Each request generated 64 tokens.

The execution order was recapture, reuse, reuse, recapture. Each stage included one warm-up cohort and three measured cohorts. This produced 48 measured responses per comparison.

Both routes in the accepted comparison used:

- `OMP_NUM_THREADS=1`.
- `MKL_NUM_THREADS=1`.
- `OPENBLAS_NUM_THREADS=1`.
- Fast dispatch and `TT_METAL_DISABLE_SFPLOADMACRO=1`.
- A 1x2 mesh, a sequence limit of 2048, and a generation limit of 64.
- A 900-second server timeout, a 30-second termination grace, and a 96 GiB memory limit without swap.
- Read-only model weights and tensor caches, private compiler scratch space, and loopback HTTP access.

The inner Qwen package remained `/nix/store/qw9llks4k65np226xygfrmzx1qcbs6nz-qwen36-0.1.0`. The wrapper does not rebuild or modify that runtime.

## Accepted measurements

| Median metric | Recapture | Reuse |
| --- | ---: | ---: |
| Request time | 7.752 s | 7.203 s |
| Trace preparation | 591.96 ms | 0.728 ms |
| Time to first token | 2.704 s | 2.733 s |
| Decode rate per sequence | 15.278 tokens/s | 15.258 tokens/s |
| Tail token latency | 68.27 ms | 67.75 ms |

The request-time ratio was 1.076. Decode rate remained effectively unchanged. This optimization removes trace preparation between requests. It does not make the decode kernels faster.

Repeated-stage ratios were 1.086 for recapture and 1.089 for reuse. Both met the existing maximum of 1.25. Every measured reuse avoided recapture, retained one resident trace, and reported stable bindings and clean resets.

The [accepted receipt](evidence/tt-metal-0.77/qwen-trace-reuse/host1/receipt.json) records `admitted: true`. That verdict covers this diagnostic workload. The repository's production admission binding remains false.

## Rejected run and recovery

The first comparison used the inherited host environment. Its outputs matched, but repeated-stage ratios of 1.947 and 1.439 failed the stability gate. The [rejected receipt](evidence/tt-metal-0.77/qwen-trace-reuse/default-threads/receipt.json) remains available.

A separate check of the packaged Python environment reported 32 Torch threads by default and one with the explicit limits. The later comparison met the stability bounds. This does not prove that thread count alone caused the earlier variance.

The first thread-limited attempt stopped before measurements. Pueue could not save its state and terminated its tasks during a disk-space incident. Its detached probe survived. The agent sent SIGTERM and verified clean driver closure.

One retry used a separate timed user service and memory-backed logs. All four stages completed and closed the drivers. The work used two complete comparisons and one interrupted startup. No global cleanup or unrelated service change occurred.

## Opt-in command

```sh
nix run .#qwen36-trace-reuse-host1 -- --help
```

The executable is `qwen36-p150x2-trace-reuse-host1-admission`. Here, `host1` means one host thread, not batch width one. The command still requires the existing B=4 diagnostic workload.

The wrapper fixes the three thread limits, disables SFPLOADMACRO, and clears the slow-dispatch override. It delegates to the existing trace-reuse admission command. It does not change the production binding, fusion controls, MTP policy, or model source.

Use a bounded, read-only harness for hardware runs. Do not substitute this diagnostic command for the production service.

## Validation and search limits

- The existing Qwen package checks and basic simulator smoke passed before the experiment.
- The receipt checker passed its positive and negative self-tests.
- Replay from the saved response files reproduced the accepted receipt byte for byte.
- The new package passed its help check with invalid inherited thread settings. It rejected an unknown option and eager mode without device access.
- The new package check and existing Qwen package check passed after packaging.

Trace reuse was the validated approach. Explicit host limits defined the accepted test configuration. Their isolated performance effect remains unproven.

A second source check rejected the assumption that the decode synchronization call was obviously redundant: `TRACE_BLOCKING_REPLAY` is false. No synchronization change ran. Fused prefill and MTP changes remained outside this round.

The checks do not establish B=1 acceleration, arbitrary continuous batching, slot recycling, sampled generation, or broad model accuracy. The numerical simulator limitation remains unchanged. Nothing was deployed or pushed.
