# P150x2 throughput baseline analysis

This record binds the baseline used by the throughput change. It does not make a performance claim.

## One-card behavior

The pre-change runtime had no one-card mode. `DeviceContext` always opened a `1x2` mesh with physical device IDs 0 and 1. Its tensor checks required exactly two replicated shards. A valid one-card throughput baseline was therefore unavailable.

The optimized runtime adds an explicit one-card worker. Physical measurements must still prove its behavior before admission.

## Two-card behavior

The model-target profile identifies the pre-change topology as `full-lockstep`. The runtime replicated every host tensor across the `1x2` mesh. Each decode step required both cards to select the same token. Both cards therefore served one request together instead of serving independent requests.

## Retained physical attempt

`fixtures/terminal-attempt-3/` retains the bounded physical attempt from 2026-07-18.

- The process opened physical devices 0 and 1.
- TT-Metal enabled the device profiler and Watcher.
- Blackhole BRISC firmware exceeded its code region before model execution.
- No model operation, launch-count, latency, or throughput profile was produced.
- Owner restoration did not reach its health window.

The failure prevents a measured bottleneck or fusion claim. This change therefore selects the smallest source-proven bottleneck removal: replace lockstep replication with one complete replica per card, and remove evidence work from timed decode. It does not add a custom fused operator.

## Simulator gate

The packaged Blackhole `1x1` smoke passed on 2026-08-08. `fixtures/simulator-smoke-valid.json` records one device open, one RWKV layer, two recurrent steps, FP32 device state, and 32 matrix multiplications. The receipt is simulated and makes no performance claim.

## First physical candidate

`fixtures/physical-throughput-first-candidate/` retains the first complete five-repeat aggregate receipt. Every short and long stream matched the pinned FP32 CPU reference. The candidate passed aggregate speedup at `1.6663084339651104` but failed p99 admission at `0.6458061323139735`.

One profiled dual worker stalled. TT-Metal also reported full profiler buffers and dropped markers. The deterministic decision was to separate profiler attestation from timed repeats. This result did not justify a fused operator.

## Admitted physical result

`fixtures/physical-throughput-admitted/` retains the admitted aggregate evidence. Matching profiler preflight runs attributed physical devices 0 and 1. The five timed repeats did not enable profiler instrumentation.

- One-card throughput: `21.289619409251138` tokens/s.
- Two-card throughput: `38.65481160499171` tokens/s.
- Speedup: `1.8156647548238811`.
- Scaling efficiency: `0.9078323774119406`.
- P99 latency change: `-0.2225636226581016`.

All short and long generated streams matched the pinned FP32 CPU reference exactly. The result passes the `1.50` speedup bound and the `0.10` p99 regression bound.

## Physical profiler bottleneck

A bounded streaming reduction of the first physical one-card profiler CSV paired zone start and end records without loading the file into memory. `TRISC-FW` had `747815417` aggregate cycles, and `TRISC-KERNEL` had `482984564` aggregate cycles. The next surfaces were `BRISC-FW` at `343672436` and `NCRISC-FW` at `251997019` cycles.

The largest observed steady-state surface is therefore Tensix compute execution, not host readback or receipt work. Dropped profiler markers prevent a complete operation-level ranking. This evidence does not justify a custom fused kernel.

## Runtime batching investigation

A source audit found three fast-path defects:

- The scheduler advanced each request separately. It did not execute a card-local batch.
- Every prompt token executed final normalization and the full vocabulary head.
- Every selected token caused a blocking host read before the next device step.

`fixtures/physical-throughput-batched-candidate/` retains the first fixed-batch aggregate receipt. The candidate kept exact FP32 token parity. It measured `49.88481963465173` tokens/s on one card. This is `134.31522506679903%` above the retained admitted one-card result.

The same candidate measured `50.67726546622957` tokens/s on two cards. This is `31.102088878594714%` above the retained admitted two-card result.

The candidate did not pass scaling admission. Its speedup was `1.0158855106098728`, and its p99 change was `0.25554955976220295`.

The four-request workload now fills one four-row batch on one card. The two-card run gives each card only two rows. Therefore, this workload does not measure saturated scaling after batching.

Do not weaken the scaling gate to admit this result. A new scaling workload must saturate each card without exceeding four live state slots.

## Physical B1 decode candidate

`fixtures/physical-b1-decode-candidate/` retains a five-repeat B1 measurement. One physical P150 card generated 64 deterministic tokens from prompt token `0`.

Mean throughput was `10.318388240286174` tokens/s. Median throughput was `8.087155632081794` tokens/s. Samples ranged from `6.675900729440436` to `14.994682237861321` tokens/s.

Card 0 ended at `800 MHz` and `66.4°C`. The large range did not coincide with evident thermal throttling.

Selected-token readback remains a measured optimization target. B1 performs a blocking read after every token. Device-resident continuation must retain deterministic token evidence without timing each readback.

This candidate has no profiler attribution or pinned FP32 CPU-reference comparison. It cannot satisfy physical scaling admission.

## Device-continuation investigation

`fixtures/physical-b1-decode-candidate/device-continuation-investigation.md` records a reverted device-resident token experiment. All B1 token streams stayed deterministic.

An alternating five-pair A/B test did not prove a sustained gain. Aggregate throughput changed by `-7.325042354322009%`. Mean throughput changed by `0.11850828141355407%`, while median throughput improved by `14.030194778215389%`.

Further isolation found a physical process-isolation defect. Each single-card worker used the dual-card descriptor, so both UMD processes discovered and opened both cards.

Each worker now selects its card with `TT_VISIBLE_DEVICES`. It uses the single-card P150 descriptor and opens filtered logical device 0.

Five concurrent control samples kept exact tokens on both cards. A complete strict benchmark also passed profiler attribution and exact FP32 token parity.

`fixtures/physical-throughput-filtered-workers-candidate/` retains that rejected result. One-card throughput was `51.61843281609733` tokens/s. Two-card throughput was `42.20120195168836` tokens/s.

The speedup was `0.8175606977848389`, and the p99 change was `3.4918677732695094`. The B4-versus-B2 workload still does not prove saturated scaling.

The bounded scheduler now supports at most two waves. The saturated profile gives one-card mode two B4 waves. Two-card mode runs one B4 wave per card.

`fixtures/physical-throughput-saturated-admitted/` retains the admitted result. One-card throughput was `52.25896285041297` tokens/s. Two-card throughput was `109.66330232771391` tokens/s.

The speedup was `2.0984592182132698`. Scaling efficiency was `1.0492296091066349`, and the p99 change was `-0.053870298117255126`.

All profiler and timed token streams matched the pinned FP32 CPU reference. The result passes both physical admission bounds.

Profiler inspection then found `8,463` tiled reshape launches in one 39-step B4 worker. Reusing BF16 head rows reduced this count to `6,591` without changing other leading kernel signatures.

`fixtures/physical-throughput-row-reuse-admitted/` retains the optimization evidence. Five alternating A/B pairs measured a `3.666199109471613%` aggregate gain. Mean and median gains were also positive.

The following strict run reached `60.59247374263283` tokens/s on one card and `123.79287232979172` tokens/s on two cards. Its speedup was `2.0430404088732743`.

All strict token streams stayed exact. Separate strict runs remain too nonstationary to isolate the optimization gain, so the claim uses the alternating A/B result.

Static correction vectors now load in head-row shape. Group-normalization scale and bias stay in that shape through attention correction.

`fixtures/physical-throughput-head-row-normalization-admitted/` retains this result. It removed another `1,404` tiled reshape launches and measured a `9.063146955374467%` aggregate A/B gain.

The final strict run reached `62.69137518901232` tokens/s on one card and `125.85112395302231` tokens/s on two cards. Its speedup was `2.007471100667126`.

The continuation experiment remains reverted. Current evidence does not justify replacing blocking readback.

## Admission boundary

Only `rwkv7-p150x2-throughput-benchmark` can produce candidate physical scaling evidence. `rwkv7-p150x2-throughput` recomputes the metrics and rejects:

- missing physical or profiler attribution;
- estimated or simulated runs;
- a dual run that does not assign requests to both cards;
- mismatched one-card and two-card workloads;
- state readback, profiler flush, control replay, or artifact writes inside timing;
- speedup below `1.50`; or
- p99 latency regression above `0.10`.
