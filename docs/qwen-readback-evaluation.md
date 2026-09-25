# Qwen readback evaluation

## Result

The queued-read candidate did not improve the measured readback time. Its median was 121.35 microseconds, against 119.75 microseconds for the baseline. The baseline/candidate speed ratio was 0.987.

The candidate passed correctness checks, but it did not establish a performance benefit. The runtime change and its candidate-only tests were removed. The package returned to `/nix/store/acqd15fjnx3gcilph1vc80q7qkqw2m9h-qwen36-0.1.0`.

The previous binding optimization remains intact. No new runtime optimization, deployment, or push occurred in this round.

## Readback experiment

The baseline converts indices and values through two blocking reads. The candidate starts the index read without blocking, then performs a blocking value read on the same queue.

The candidate retains both host buffers until completion. On a transfer error or interruption, it drains that queue before returning the error. It does not remove the full-mesh barrier after trace replay.

The package-matched TT-Metal 0.77 implementation supports this ordering:

- `ttnn/ttnn/operations/core.py`, `to_torch`, reads device tensors through `from_device`.
- `ttnn/cpp/ttnn/operations/core/core.cpp`, `from_device`, calls `tensor.cpu`.
- `ttnn/core/tensor/tensor_ops.cpp` enqueues the read on the supplied queue.
- `tt_metal/distributed/fd_mesh_command_queue.cpp`, `submit_memcpy_request`, calls `finish_nolock` for a blocking read.
- `finish_nolock` waits for outstanding reads on that queue.

The [receipt](evidence/tt-metal-0.77/qwen-readback/receipt.json) records a physical 1x2 mesh with two command queues. Both reads use queue zero. Each sample contains 1,000 readback calls, in baseline, candidate, candidate, baseline order.

The synthetic fixture uses row-major tensors with four winners per device. It checks ties, buffer updates at unchanged addresses, invalid tokens, and retry after a rejected second transfer. It does not reproduce the full-model output layout or measure end-to-end decode.

The diagnostic source was `pkgs/qwen36/readback-probe.py`. It was removed on 2026-09-25, when the greedy trace tail began combining the shard winners on device (`.cairn/changes/stream-qwen38-greedy-decode`); the two-read readback it compared no longer exists. The [rejected patch](evidence/tt-metal-0.77/qwen-readback/rejected-candidate.patch) preserves the candidate and its ordering tests. It is not part of the package patch list.

Candidate package checks passed. Tests covered transfer order, host-buffer lifetime, first-read errors, second-read errors, interruption, tie selection, and vocabulary rejection. Baseline package checks passed again after removal of the candidate.

## Full-model component profile

The remaining model-probe budget ran the existing B=1 component profiler on the restored baseline. It used the unchanged `decode-component-profile.ncl`, five samples per group, and one warm-up replay.

The [component receipt](evidence/tt-metal-0.77/qwen-readback/component-profile.json) records these host-timed measurements:

| Measurement | Median or weighted subtotal |
| --- | ---: |
| Full trace replay | 54.34 ms |
| Host input update | 0.470 ms |
| Winner readback | 0.695 ms |
| GDN mixer, weighted across 48 layers | 26.34 ms |
| MLP, weighted across 64 layers | 23.86 ms |
| Full-attention mixer, weighted across 16 layers | 5.65 ms |
| LM head and argmax | 2.07 ms |

Weighted subtotals extrapolate representative layers. They are not direct measurements of every layer. Their sum/full-replay ratio was 1.067, within the unchanged closure limits.

The evaluator rejected attribution and selected no component:

- Input-update spread was 1.25665, above the 1.25 limit.
- The largest weighted subtotal exceeded the next by about 1.10, below the required 1.25 dominance ratio.

The raw GDN and MLP measurements are leads, not accepted target selection. No stability or dominance limit changed. This B=1 profile does not establish B=4 performance or a model speedup.

The first decode token was 11751. All full-replay samples returned token 13. The receipt records successful cleanup and `admission_eligible: false`.

The receipt also contains the legacy sentence `Production remains eager.` It is not deployed-state evidence. The actual production service remains inactive.

## Bounds and remaining work

This round used one 300-second readback probe and one 900-second component probe. Both finished within their limits and closed the drivers. Both used private scratch space and explicit memory limits. Model weights and caches remained read-only.

The basic simulator smoke and Qwen package checks passed before the candidate change. These checks do not substitute for the recorded physical measurements.

Two approaches remain distinct:

- Queued readback: correctness passed, measured speedup absent, runtime change removed.
- Full-mesh barrier removal: not attempted, because cross-queue ordering still needs a separate proof.

The component profile supplies new evidence but fails its selection rules. The next useful check is B=4 attribution for GDN and MLP, without treating either as an accepted dominant target.
