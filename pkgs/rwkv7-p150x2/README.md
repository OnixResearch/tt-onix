# RWKV-7 P150x2 runtime

This package has two separate execution paths.

## Evidence mode

The default `--tokens` mode preserves the original full-device parity path. It opens an exact `1x2` mesh on physical devices 0 and 1. It retains tensor readback, control replay, per-stage profiler ranges, and artifact writes.

Do not use evidence-mode elapsed time as an inference-throughput result.

## Fast worker mode

`--fast-worker` opens one exact physical card as a `1x1` mesh. Each worker loads one complete model replica. The benchmark launcher runs card 0 and card 1 workers at the same time. Requests do not cross cards.

Each worker sets `TT_VISIBLE_DEVICES` to its physical card. It uses the pinned single-card P150 descriptor and opens filtered logical device 0. This isolation prevents one worker from reconfiguring the other card.

The worker keeps one static batch per card. Each batch row has independent FP32 WKV state and token-shift state. The worker uses deterministic, bounded state slots.

The saturated benchmark supports at most two process waves. It never assigns more active state slots to one worker than the admitted card-local capacity.

## Card-local capacity plan

Nickel owns the typed capacity candidates B4, B16, B32, and B64 (`throughput-capacity-profile.ncl` in the evidence package). The pure planner (`rwkv7-p150x2-throughput capacity-plan`) validates every candidate device-free: it derives FP32 recurrent-state bytes, BF16 token-shift bytes, temporary tensor estimates, and a safety reserve with checked arithmetic, and it rejects overflow or an insufficient device budget before any device opens.

The benchmark launcher writes the admitted capacity and its checked memory bounds into each worker request. The worker re-runs the same checked derivation before opening its device. It records the effective batch size, queue occupancy, memory plan, per-request latency, and host gap (host-only batch-step scheduling time) in its receipt.

A capacity above four becomes the default only after simulator, exact-parity, stable A/B, and strict two-card physical evidence pass. See the change `cairn/changes/scale-rwkv7-card-local-batching/`.
The WKV path reuses BF16 `r`, `k`, and `v` head rows during attention correction. Separate FP32 cast views preserve recurrent-state precision.

Static group-normalization and `r_k` vectors load in head-row shape. Attention correction stays in that shape until the output projection.

WKV matrix products use the transpose flags of `ttnn::matmul`. The runtime does not dispatch separate vector-transpose operations in the decode loop.

Time mixing and channel mixing use the tensor form of `ttnn::lerp`. This fused kernel replaces the subtraction, multiplication, and addition sequence.

The value, decay, and attention-gate projections pass their static biases to `ttnn::linear`. This removes separate row-broadcast additions.

The FP32 WKV output matmul writes BF16 directly for attention normalization. This removes one recurrent output typecast per layer and token.

The same matmul consumes the BF16 read vector directly. The pinned workload preserves exact tokens while removing another vector typecast per layer and token.

The worker omits final normalization and the vocabulary head for nonfinal prompt tokens. It reads one selected-token vector after each required batched head.

The fast path does not retain per-operation event strings or maps. Profiler attribution remains available through the TT-Metal device profiler.

The timed region contains warmed prompt and decode operations. It excludes:

- recurrent-state readback;
- control replay;
- profiler flush;
- receipt construction; and
- artifact writes.

The worker flushes the device profiler after timing.

## Simulator smoke

Run the bounded Blackhole `1x1` smoke before a new physical attempt:

```sh
"$EVIDENCE/bin/rwkv7-p150x2-simulator-smoke" /absolute/new/simulator-root
```

The script uses packaged `libttsim_bh.so`, slow dispatch, and a named 60-minute timeout. It opens one simulated device, loads the model weights, executes RWKV layer 0 twice, and carries the FP32 recurrent matrix between steps. Its receipt is marked simulated and cannot satisfy physical throughput admission.

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

Set `EVIDENCE` to the built `rwkv7-p150x2-evidence` store path. The benchmark first runs matching one-card and two-card profiler attestations. It then runs five unprofiled one-card samples and five unprofiled two-card samples.

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
