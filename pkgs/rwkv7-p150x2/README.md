# RWKV-7 P150x2 runtime

This package has two separate execution paths.

## Evidence mode

The default `--tokens` mode preserves the original full-device parity path. It opens an exact `1x2` mesh on physical devices 0 and 1. It retains tensor readback, control replay, per-stage profiler ranges, and artifact writes.

Do not use evidence-mode elapsed time as an inference-throughput result.

## Fast worker mode

`--fast-worker` opens one exact physical card as a `1x1` mesh. Each worker loads one complete model replica. Requests do not cross cards.

After initialization, each worker signals readiness. The launcher then starts warmup on all active workers.

After warmup, each worker signals timed readiness. The launcher releases the timed start only after all workers reach this second barrier.

Worker receipts include monotonic start and end times. The launcher rejects start skew above 10 milliseconds and measures the full synchronized wave interval.

The launcher disables TT-Metal Inspector for benchmark workers. This prevents Inspector RPC collisions and Inspector artifact writes during timed execution.

Each worker sets `TT_VISIBLE_DEVICES` to its physical card. It uses the pinned single-card P150 descriptor and opens filtered logical device 0. This isolation prevents one worker from reconfiguring the other card.

The worker keeps one static batch per card. Each batch row has independent FP32 WKV state and token-shift state. The worker uses deterministic, bounded state slots.

The saturated benchmark supports at most two process waves. It never assigns more active state slots to one worker than the admitted card-local capacity.

## Card-local capacity plan

Nickel owns the typed capacity candidates B4, B16, B32, and B64 (`throughput-capacity-profile.ncl` in the evidence package). The pure planner (`rwkv7-p150x2-throughput capacity-plan`) validates every candidate device-free: it derives FP32 recurrent-state bytes, BF16 token-shift bytes, temporary tensor estimates, and a safety reserve with checked arithmetic, and it rejects overflow or an insufficient device budget before any device opens.

The benchmark launcher writes the admitted capacity and its checked memory bounds into each worker request. The worker re-runs the same checked derivation before opening its device.

The receipt records batch size, queue occupancy, memory bounds, per-request latency, host gap, and synchronized timing bounds.

A capacity above four becomes the default only after simulator, exact-parity, stable A/B, and strict two-card physical evidence pass. See the change `cairn/changes/scale-rwkv7-card-local-batching/`.

The 2026-08-09 stalled-wave investigation re-examined the B16 block. Per-request p99 wall latency was the wrong cross-capacity metric; per-token time is the honest one. Under that gate, B16 is admitted: it is `2.7-3.7x` faster in aggregate and its p99 per-token time is `4.2-5.5 ms` versus B4's `7.3-57 ms`. The order-balanced evidence is retained in `../rwkv7-p150x2-evidence/fixtures/physical-capacity-b16-admitted/`. The residual 16-wide kernel stall remains a tracked follow-up.
The WKV path reuses BF16 `r`, `k`, and `v` head rows during attention correction. Separate FP32 cast views preserve recurrent-state precision.

Static group-normalization and `r_k` vectors load in head-row shape. Attention correction stays in that shape until the output projection.

WKV matrix products use the transpose flags of `ttnn::matmul`. The runtime does not dispatch separate vector-transpose operations in the decode loop.

Time mixing and channel mixing use the tensor form of `ttnn::lerp`. This fused kernel replaces the subtraction, multiplication, and addition sequence.

The value, decay, and attention-gate projections pass their static biases to `ttnn::linear`. This removes separate row-broadcast additions.

The FP32 WKV output matmul writes BF16 directly for attention normalization. This removes one recurrent output typecast per layer and token.

The same matmul consumes the BF16 read vector directly. The pinned workload preserves exact tokens while removing another vector typecast per layer and token.

The worker omits final normalization and the vocabulary head for nonfinal prompt tokens. It reads one selected-token vector after each required batched head.

The fast path does not retain per-operation event strings or maps. Profiler attribution remains available through the TT-Metal device profiler.

## Fused WKV candidate

The package contains a Blackhole-only fused WKV candidate. It supports 12 heads, width 64, and batches from 1 through 16.

The candidate reads BF16 vectors and FP32 recurrent state. It writes separate BF16 attention output and FP32 next state.

The device kernel includes recurrence, per-head group normalization, the `r_k` correction, and gate multiplication. The later output projection stays outside.

Each `(request, head)` work item has one deterministic core owner. Each state page has one writer, and requests never share state.

Production dispatch remains closed. `generic_wkv_step` remains authoritative until physical parity and performance admission pass.

The standalone `--fused-wkv-probe` runs the candidate at batch 16 for two recurrent steps. It does not enable production dispatch.

Physical B16 evidence shows stable one-card and two-card paired gains. Fused throughput was `437.98668576683207` tokens/s on one card. It was `722.4472805047268` tokens/s on two cards. Exact generated-token parity passed.

Synchronized fused scaling was `1.6494731551938797`, above the required `1.5`. Strict admission remains blocked. The p99 latency regression was `0.24129836831923815`; the profile permits at most `0.10`. Simulator results cannot supply physical performance evidence.

The timed region contains warmed prompt and decode operations. It excludes:

- recurrent-state readback;
- control replay;
- profiler flush;
- receipt construction; and
- artifact writes.

The worker flushes the device profiler after timing.

### Fused WKV stall capture

Use the bounded stall mode to compare synchronized, dual-card fused samples:

```sh
PROFILE="$EVIDENCE/share/rwkv7-p150x2-throughput/fused-wkv-stall-profile.json"
WORKLOAD="$EVIDENCE/share/rwkv7-p150x2-throughput/fused-wkv-stall-workload.json"
"$EVIDENCE/bin/rwkv7-p150x2-throughput-benchmark" \
  --fused-wkv-stall-capture SAMPLE_COUNT \
  "$PROFILE" "$WORKLOAD" /absolute/new/stall-capture-root
```

The sample count must be from 1 through 16. The diagnostic profile uses one warmup token and one timed token.

Each stall-probe worker sets `TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT=40000`. This value is 40 times the TT-Metal default. It gives prompt setup and timed decode a bounded device-profiler buffer.

The command rejects every sample that reports dropped profiler markers.

Each sample records step wall time and Unix time bounds in memory. These bounds correlate the timed region with TT-Metal synchronization logs.

The diagnostic worker also records model-submission time and each layer-submission time. Normal workers do not collect these boundary timings.

For selected steps, the worker separates the blocking device readback from host materialization and validation.

The final receipt compares the fastest and slowest valid samples. It includes card ownership, boundary timing, zone cycles, and program signatures.

An eight-sample physical capture found a `3.521899865365418` slow wave. Model-submission time increased by `3.45-3.53x` on both cards.

Blocking readback changed by only `0.98-1.27x`. Device profiler-zone cycles fell to `0.90x` of the normal sample.

The measured boundary is the host-visible return from each layer submission. The capture does not identify a smaller TT-NN or TT-Metal call.

The summarizer deletes each raw profiler CSV. This mode is diagnostic only. It cannot enable production admission.

## Simulator smoke

Run the bounded Blackhole `1x1` smoke before a new physical attempt:

```sh
"$EVIDENCE/bin/rwkv7-p150x2-simulator-smoke" /absolute/new/simulator-root
```

The script uses packaged `libttsim_bh.so`, slow dispatch, and a named 60-minute timeout. It opens one simulated device, loads the model weights, executes RWKV layer 0 twice, and carries the FP32 recurrent matrix between steps. Its receipt is marked simulated and cannot satisfy physical throughput admission.

Run the same layer with the fused WKV candidate explicitly required:

```sh
"$EVIDENCE/bin/rwkv7-p150x2-fused-wkv-layer-simulator-smoke" \
  /absolute/new/fused-wkv-layer-simulator-root
```

This smoke fails if the fused capability contract rejects the layer. It reads the FP32 matrix after both steps and compares the carried result with a reset-matrix control. All states must be finite, and the carried state must differ from the control. Production admission remains closed.

The retained receipt is `cairn/changes/fuse-rwkv7-wkv-kernel/evidence/phase2-fused-wkv-layer-simulator-receipt.json`. Its BLAKE3 is `eb25c84e9b8eec860bdf7649c9c59a0b98d322dd2a950cd85b5008a069574f6d`.

Run the standalone fused candidate with the packaged smoke script:

```sh
"$EVIDENCE/bin/rwkv7-p150x2-fused-wkv-simulator-smoke" \
  /absolute/new/fused-wkv-simulator-root
```

This smoke uses batch 16, all 12 heads, and two recurrent steps. It compares output and next state with the corrected fixture.

## Benchmark and admission

Build the runtime and evidence package:

```sh
nix build .#rwkv7-p150x2-runtime .#rwkv7-p150x2-evidence
```

Run the physical benchmark only through the host's owner-isolation procedure:

```sh
PROFILE="$EVIDENCE/share/rwkv7-p150x2-throughput/throughput-profile.json"
WORKLOAD="$EVIDENCE/share/rwkv7-p150x2-throughput/throughput-workload.json"
"$EVIDENCE/bin/rwkv7-p150x2-throughput-benchmark" \
  "$PROFILE" "$WORKLOAD" /absolute/new/evidence-root
```

Use the explicit candidate mode for fused-versus-generic physical evidence:

```sh
FUSED_PROFILE="$EVIDENCE/share/rwkv7-p150x2-throughput/fused-wkv-physical-profile.json"
FUSED_WORKLOAD="$EVIDENCE/share/rwkv7-p150x2-throughput/fused-wkv-physical-workload.json"
"$EVIDENCE/bin/rwkv7-p150x2-throughput-benchmark" \
  --fused-wkv-candidate "$FUSED_PROFILE" "$FUSED_WORKLOAD" \
  /absolute/new/ab-evidence-root
```

The fused profile runs three A/B pairs at the admitted B16 capacity. Candidate mode alternates generic and fused runs for each sample. It checks every generated token against the pinned FP32 reference. It retains profiler samples for both paths. This mode does not open production admission.

Set `EVIDENCE` to the built `rwkv7-p150x2-evidence` store path. The benchmark first runs matching one-card and two-card profiler attestations. It then runs the unprofiled one-card and two-card samples from the selected profile.

The benchmark streams each raw profiler CSV into a bounded card-attribution summary. It retains the BLAKE3 hash, byte count, row count, worker logs, request, and receipt. It removes the large raw CSV after validation. Profiler instrumentation does not enter the timed repeats.

Each worker also returns greedy token IDs. The launcher rejects a short or long stream that differs from the pinned FP32 CPU reference in `throughput-workload.json`.

The benchmark fails unless all conditions are true:

- every run uses physical hardware and retained profiler attribution;
- one-card and two-card samples use the same request and token workloads;
- both cards serve distinct requests in every two-card sample;
- aggregate two-card throughput speedup is at least `1.50`; and
- p99 request-latency regression is at most `0.10`.

The admitted 2026-08-08 physical result was `21.289619409251138` tokens/s on one card and `38.65481160499171` tokens/s on two cards. The speedup was `1.8156647548238811`. The p99 latency change was `-0.2225636226581016`. See `../rwkv7-p150x2-evidence/fixtures/physical-throughput-admitted/`.

The best strict result reached `86.8422965099119` tokens/s on one card and `181.8205337965597` tokens/s on two cards. Its speedup was `2.0936863844429454`. See `../rwkv7-p150x2-evidence/fixtures/physical-throughput-linear-bias-admitted/`.

Direct BF16 WKV output removed `468` more launches. Stable dual-card A/B rounds gained `5.203905022531408%`. Later strict runs passed admission but contained stalls, so they do not replace the absolute record. See `../rwkv7-p150x2-evidence/fixtures/physical-throughput-bf16-wkv-output-admitted/`.

Passing the BF16 read vector directly removed another `468` launches. Five stable card-0 pairs gained `4.651337407376599%`. See `../rwkv7-p150x2-evidence/fixtures/physical-throughput-mixed-r-admitted/`.

The pure admission command can recompute a retained receipt:

```sh
"$EVIDENCE/bin/rwkv7-p150x2-throughput" admit \
  "$PROFILE" raw-throughput-evidence.json throughput-receipt.json
```

## Non-claims

A package build or device-free fixture does not prove physical throughput, latency, thermal behavior, or profiler attribution. Simulator output cannot satisfy physical throughput admission. See `../rwkv7-p150x2-evidence/baseline-analysis.md` for the retained baseline blocker.
