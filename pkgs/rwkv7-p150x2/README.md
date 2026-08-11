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

Production fused dispatch requires the measured host scheduler contract. Card 0 uses CPU 2, card 1 uses CPU 6, and both use FIFO priority 1.

If this contract is absent, production uses `generic_wkv_step`. Explicit `generic-reference` workers also keep the authoritative fallback available for A/B checks.

The standalone `--fused-wkv-probe` runs the candidate at batch 16 for two recurrent steps. It does not enable production dispatch.

The final B16 evidence passed exact generated-token parity and all strict performance gates. Fused throughput was `502.27251306237105` tokens/s on one card.

Two-card fused throughput was `954.5668543729688` tokens/s. Scaling was `1.9004959052068073`, and p99 regression was `0.06701184998181789`.

All three one-card and two-card fused-versus-generic pairs improved. A later production-policy smoke confirmed admitted fused dispatch on both cards.

Simulator results do not supply physical performance evidence.

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

An isolated diagnostic runtime records bounded TT-Metal fetch-queue counters. The production TT-Metal package does not include these counters.

The worker also records process and thread CPU time for model and layer submission.

For selected steps, the worker separates the blocking device readback from host materialization and validation.

The final receipt compares the fastest and slowest valid samples. It includes card ownership, boundary timing, zone cycles, and program signatures.

An eight-sample physical capture found a `3.521899865365418` slow wave. Model-submission time increased by `3.45-3.53x` on both cards.

Blocking readback changed by only `0.98-1.27x`. Device profiler-zone cycles fell to `0.90x` of the normal sample.

The measured boundary is the host-visible return from each layer submission. The capture does not identify a smaller TT-NN or TT-Metal call.

A later complete capture falsified fetch-queue backpressure. Extra fetch-queue wait covered only `0.028-0.057%` of the slow-wave wall-time increase.

A CPU-time capture found a `1.647x` slow wave. Model-submission process CPU time increased by `1.640-1.647x`.

Model-submission thread CPU time increased by `1.623-1.625x`. The stall is on-CPU host work, not device work or an off-CPU queue wait.

A root eBPF capture found stable call counts across eight waves. Measured TT-NN and TT-Metal subphases all expanded by about `2.3-3.0x`.

`enqueue_mesh_workload` remained about `18%` of total time in both selected waves. No smaller blocking host call dominates the slowdown.

Additional off-CPU time explained only `6.0-10.4%` of the excess.

A later eight-sample hardware-counter capture found a `2.38536978179575x` slow wave. Instructions changed by only `1.003-1.009x`.

Task-clock counts increased by `2.209-2.228x`. IPC fell to `0.490-0.497x`, and instruction rate fell to `0.450-0.457x`.

Average core frequency fell by about `8%`. Cache misses increased by `2.120-2.155x`, while cache references increased by only `1.204-1.221x`.

Context switches increased by `5.41-11.17x`. CPU migrations increased by `14.16-15.79x`.

The stall is lower host CPU throughput, not extra instructions.

A primary-thread-only test pinned cards 0 and 1 to CPUs 2 and 6. All measured CPU migrations fell to zero.

The maximum wave ratio fell from `2.38536978179575` to `1.1594243666397046`. Median wall time fell by `19.39%`.

Wall-time standard deviation fell from `0.2604570536328125` to `0.036949182658166574` seconds. Process-wide TT-Metal threads remained unpinned.

A split-CCD test moved card 1 to CPU 10. Its maximum wave ratio increased to `2.339356316019673`.

Median wall time increased by `15.85%`. Standard deviation increased by `7.62x`, so the split-CCD layout was rejected.

A full same-CCD A/B rerun reached `1.7248059373131326x` dual-card fused scaling. It still failed the p99 gate at `0.381053554753073`.

One dual fused pair lost to generic at `0.9091285937922943x`. Primary-thread pinning alone is not sufficient for production admission.

A helper-thread exclusion control produced a rejected `4.785382972`-second wave. It had 1,736 context switches and incomplete profiler pairing.

The helper-thread exclusion code was removed.

A bounded root test then used FIFO priority 1 for only the pinned primary threads. All eight waves passed complete profiler validation.

The maximum wave ratio fell to `1.0865249230572205`. Instructions changed by less than `0.22%`, and migrations remained zero.

The FIFO mode restores normal scheduling after timed decode. It requires root or `CAP_SYS_NICE`.

The summarizer deletes each raw profiler CSV. This mode is diagnostic only. It cannot enable production admission.

## Persistent decode

The fast worker has an admitted, bounded Metal Trace path for homogeneous B4 workloads. The closed decode-window set is `[2, 4]`.

Each request in a batch must use the same prompt length and the same admitted window. Other window sizes use normal dispatch.

The worker processes each prompt prefix before the trace. The trace consumes the final prompt token and chains the selected number of device tokens.

One `execute_trace` call submits the complete decode region. Token selection and FP32 recurrent-state carry stay on the device inside that region.

Before trace capture, the worker runs the same request through normal dispatch. It requires exact generated-token and FP32 state BLAKE3 parity.

The worker reads outputs only after trace replay. It then acknowledges completion, zeros all state tensors in place, and releases each slot.

Pre-replay cancellation, timeout, and failure use the same acknowledgement, zeroing, and release sequence.

All token, state, slot, weight, program, runtime, checkpoint, output, and profile bindings use BLAKE3 identities.

The installed Nickel profile sets `physically_admitted = true` after all retained gates passed. Eligible requests use persistent trace by default.

Normal dispatch remains the fail-closed path for unlisted windows, stale bindings, invalid profiles, stall probes, or failed capability evidence.

Five order-balanced four-token samples measured a `2.867805089379262x` one-card median paired speedup. Every pair exceeded `2.849279940121875x`.

The synchronized P150x2 samples measured a `3.084333977350274x` median paired speedup. Every pair exceeded `3.011763122817449x`.

Normal dispatch used four timed model submissions for the four-token window. Persistent trace used one bounded replay submission.

Set `RWKV7_PERSISTENT_DECODE_CANDIDATE=1` only to test a non-admitted profile. Set it to `0` for a timing-matched normal-dispatch A/B reference.

Omit the variable for profile-owned production selection.

### Production telemetry

Each successful fast-worker receipt contains `persistent_decode_telemetry` with schema version `1`.

The event records these bounded values:

- Eligibility and physical-profile admission.
- Selected dispatch mode and decode window.
- The exact fallback reason.
- Completion, cancellation, timeout, and failure counts.
- Trace setup, replay, and cleanup durations.
- Generated-token and FP32 recurrent-state parity status.
- Cleanup status.

The event excludes prompt text, token IDs, request IDs, filesystem paths, and binding identities. Use the detailed `persistent_decode` section for evidence inspection.

Validate the telemetry contract with this command:

```sh
"$EVIDENCE/bin/rwkv7-p150x2-persistent-decode-evidence" \
  telemetry /absolute/path/to/worker-receipt.json
```

A fatal worker error can stop receipt creation. Process supervision must report that error because this event is not a failure transport.

### Production monitoring and alerts

Collect an explicit bounded receipt batch with the installed monitoring policy:

```sh
MONITORING_POLICY="$EVIDENCE/share/rwkv7-p150x2-throughput/persistent-decode-monitoring-policy.json"
"$EVIDENCE/bin/rwkv7-p150x2-persistent-decode-monitor" \
  "$MONITORING_POLICY" \
  /absolute/path/to/worker-receipt-1.json \
  /absolute/path/to/worker-receipt-2.json \
  >persistent-decode-monitoring.json
```

The command does not scan directories or watch files. It accepts only explicit paths and reads at most the policy byte limit.

The output binds the policy and ordered event batch with BLAKE3. It contains bounded metrics and fixed alert codes. It does not contain the source paths.

The command uses these process statuses:

- Status `0`: clean batch.
- Status `2`: fallback-rate warning. Persistent admission remains retained.
- Status `3`: critical safety alert. Disable persistent admission for the affected profile.
- Status `64`: invalid policy, arguments, or bounded input.

Critical alerts cover invalid telemetry, duplicate events, parity failure, cleanup failure, replay timeout, terminal failure, and unsupported admission. The warning threshold requires eight accepted samples and a fallback rate above `0.25`.

The receipt recommends `disable` after any critical alert. The command does not mutate the installed profile. A production controller or operator must apply that recommendation.

This command is a local collector and alert classifier. It does not provide a network exporter, paging provider, daemon, or canary traffic controller.

Run the representative Blackhole simulator trace before a physical test:

```sh
"$EVIDENCE/bin/rwkv7-p150x2-persistent-decode-simulator-smoke" \
  /absolute/new/persistent-decode-simulator-root
```

The simulator receipt covers real weight loading, four traced layer steps, FP32 state carry, one trace replay submission, and state zeroing.

Metal Trace does not support slow dispatch, so this smoke uses simulator fast dispatch. It makes no determinism or performance claim.

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

This smoke fails if the fused capability contract rejects the layer. It reads the FP32 matrix after both steps and compares the carried result with a reset-matrix control. All states must be finite, and the carried state must differ from the control. Simulator evidence cannot open production admission.

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
RWKV7_PRIMARY_THREAD_CPUS=2,6 \
RWKV7_PRIMARY_THREAD_FIFO_PRIORITY=1 \
"$EVIDENCE/bin/rwkv7-p150x2-throughput-benchmark" \
  --fused-wkv-candidate "$FUSED_PROFILE" "$FUSED_WORKLOAD" \
  /absolute/new/ab-evidence-root
```

The fused profile runs three A/B pairs at the admitted B16 capacity. Candidate mode alternates `generic-reference` and fused workers. It checks every generated token against the pinned FP32 reference. It retains profiler samples for both paths. Candidate receipts cannot open production admission.

FIFO scheduling requires root or `CAP_SYS_NICE`. Production fails closed to the generic path unless CPUs 2 and 6 and FIFO priority 1 are active.

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
