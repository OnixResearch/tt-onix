# Where the B=1 decode step goes: per-op prefix timing

Recorded 2026-09-24. Physical run on devices `0` and `1` of the P150x2 mesh
(runtime `tt-metal-0.77.0`, package
`/nix/store/ylh39mq52aky0yjiiky30pavnjyip86c-qwen38-0.1.0`, B=1). The service
was isolated with `ttwkv7-owner-control`. The run is diagnostic and supports no
admission claim.

## Method

The probe loads the model through the component-probe path. It prefills
"The capital of France is" and wraps every top-level TTNN device operation. For
each prefix of the first `n` operations, it captures one trace of one component
and replays it 40 times back to back, three times. It records the median
per-replay time. The difference between consecutive prefixes is the marginal
traced cost of operation `n`, including its dispatch gap. Collective helpers
(`tt_all_reduce`, `tt_all_gather`) are atomic. Per-operation deltas carry a few
microseconds of noise. Group sums are stable. Raw rows are in
`op-prefix-b1.json`. The throwaway probe is at
`.pi/tmp/qwen38-op-prefix/probe.py`.

Full decode trace replay in the same session was 50.95 ms with the GDN layers in
the fused recurrence mode (`full`).

## Component totals (one layer, µs)

| Component | Ops | Traced cost |
| --- | ---: | ---: |
| GDN mixer, composite route (`off`, served) | 78 | 529.4 |
| GDN mixer, fused recurrence (`full`) | 74 | 459.2 |
| MLP | 9 | 362.5 |
| Full-attention mixer | 57 | 345.5 |
| LM head and shard argmax | 11 | 2063.7 |

## GDN mixer interior (µs)

| Group | `off` | `full` |
| --- | ---: | ---: |
| norm + all-gather | 34.7 | 40.3 |
| `[qkv\|z\|a\|b]` projection | 120.1 | 116.1 |
| slices of the projection | 10.8 | 9.5 |
| conv shift, multiply-accumulate, SiLU | 66.6 | 66.0 |
| q/k/v split and GQA `repeat_interleave` | 51.7 | 53.3 |
| beta and g scalars | 7.0 | 4.5 |
| typecasts | 6.2 | 8.1 |
| q/k L2 norms (+ scale) | 20.6 | 10.6 |
| recurrence (composite ops / head-major relayout + kernel) | 122.1 | 39.2 + **27.0** |
| state copy / output relayout | 7.8 | 1.6 |
| gated RMSNorm and SiLU gate | 18.8 | 12.9 |
| out projection | 42.5 | 49.6 |
| reduce-scatter | 17.2 | 17.3 |
| residual add | 3.3 | 3.3 |

## What this corrects

- The fused recurrence kernel takes **27 µs** per layer at B=1, not 549 µs. The
  0.549 ms per-layer figure used in `optimize-gdn-decode-kernel-grid-and-prep` is
  the whole mixer. It includes both projections, the collectives, and about
  230 µs of glue between the projections. The "latency-bound loop" reading in
  that change's `loop-cost-nature.md` rests on that misattribution.
- Glue between the in and out projections costs 233 µs in the fused route and
  308 µs in the composite route. That is 11 to 15 ms of a 51 to 54 ms step.
- Weight reads run at similar effective bandwidth everywhere. The `[qkv|z|a|b]`
  projection reads 44.8 MB per device in 116 to 120 µs. The MLP down projection
  reads 47.3 MB in 121 µs. BFP4 gate and up projections read 25 MB each in 75
  and 90 µs. The LM head reads 675 MB in 1.77 ms.

## Collective alternatives

A standalone traced microbenchmark on the same mesh used a `[1, 1, 32, 5120]`
bf16 partial with Linear topology and two links. It is recorded in
`.pi/tmp/qwen38-op-prefix/ccl.log`.

| Pattern | µs per call |
| --- | ---: |
| reduce-scatter only | 26.8 |
| reduce-scatter + all-gather (current decode pattern) | 40.8 |
| `ttnn.all_reduce` | 49.8 |
| all-gather of partials + slice/add | 38.1 |

No alternative saves more than about 3 µs per sublayer, so collectives are not the
lever at B=1.

A parameter sweep made the standalone all-gather 24.8 → 18.9 µs (two links, one
worker) and the reduce-scatter 26.4 → 20.2 µs (one link, one worker, one chunk
per sync). Applied in-model to every B=1 decode collective, the same settings
made the full step slower. The step measured 42.56 ms at baseline and 42.88,
42.99, and 43.19 ms for all-gather, reduce-scatter, and both, with identical
tokens. Standalone collective timings do not transfer to the model, so the
defaults stay.

## Matmul grids

A standalone sweep of the 1D-multicast decode grids (22 to 110 cores) and a
DRAM-sharded variant for every TP2 decode projection found a flat optimum. BFP8
projections run at 305 to 358 GB/s across grids, and BFP4 projections at 250
to 267 GB/s. The DRAM-sharded variant was slower for every shape. The current
program configs are within noise of the best grid.
