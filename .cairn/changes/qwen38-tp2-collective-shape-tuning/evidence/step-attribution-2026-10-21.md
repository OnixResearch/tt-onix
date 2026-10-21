# Step attribution for TP2 collectives, 2026-10-21

Baseline evidence for task t1, recorded on physical PCIe devices 0 and 1 (two P150a,
firmware 19.11.0.0, driver 2.10.0) with package
`5jk1x79sd33a5mfgdvvny8kzsbsm97ql-qwen38-0.1.0` and the `step` GDN decode route.

## What was measured

Two harnesses, both diagnostic:

1. The packaged `qwen38-p150x2-decode-component-probe` (width 1, its fixed prompt) for
   per-component medians.
2. A throwaway trace harness that captures one operation group and replays it 200 times
   per sample: the GDN mixer, one MLP forward, and one `tt_all_reduce` at widths 1 and 16.
   Trace replay removes host dispatch from the number.

## Component medians (packaged probe, width 1)

| Component | Per layer | Layers | Step total |
| --- | ---: | ---: | ---: |
| gdn_mixer | 0.3159 ms | 48 | 15.16 ms |
| mlp | 0.3894 ms | 64 | 24.92 ms |
| full_attention_mixer | 0.3705 ms | 16 | 5.93 ms |
| lm_head_argmax | 2.0842 ms | 1 | 2.08 ms |
| embedding | 0.0497 ms | 1 | 0.05 ms |
| **sum** | | | **48.14 ms** |

The measured step on this route is 44-50 ms, so the component sum is consistent with it.

## Traced groups

| Width | GDN mixer | MLP forward (no norm/add) | One all-reduce |
| ---: | ---: | ---: | ---: |
| 1 | 265.1 us | 335.3 us | 25.1 us |
| 16 | 389.7 us | 335.3 us | 30.0 us |

The MLP forward is flat in width: it is weight-bandwidth-bound (gate/up are bfloat4_b,
down is bfloat8_b). The GDN mixer grows with width through the recurrence instance loop
(the optimized kernel is 51.6 us at width 1 and 186.7 us at width 16, measured
separately).

## Collective share

A decode step issues 112 row-parallel collectives: one per MLP down-projection (64) and
one per GDN out-projection (48). At 25.1-30.0 us each that is **2.8-3.4 ms of a 45-50 ms
step, about 6-7 percent**. The collectives are not the step's dominant term.

`matmul_reduce_scatter_decode` already exists in `tp_common.py` and fuses the down-proj
matmul with the reduce-scatter. It is unused, and its own docstring records why: at decode
M=1 the 2-D matmul collapses to roughly eight cores and the fused form loses. The separate
matmul plus `reduce_scatter_minimal_async` is therefore the already-selected mechanism,
not an untried option.

## LM head: the last interleaved weight matmul

The LM head is the one large decode matmul still on an interleaved weight. Traced alone it is
**1784.7 / 1785.1 / 1788.7 / 1785.2 us** at M = 1 / 8 / 16 / 32: flat in M, so it is pure
weight streaming. The per-device weight is `[5120, 124160]` bfloat8_b, 636 MB, which gives
about **356 GB/s**. The component probe's `lm_head_argmax` (2084 us) is this matmul plus the
final norm and the two-stage argmax/value reduction.

The DRAM-sharded decode recipe the MLP and GDN projections use does not take this shape.
`create_dram_sharded_mem_config(5120, 124160)` produces eight shards of `[5120, 15520]`,
and `create_dram_sharded_matmul_program_config(1, 5120, 124160)` then requests
**9,046,592 B of static circular buffers** against the 1,572,864 B L1 limit, so the program
fails to compile before any timing. The candidate is rejected on that bound; a sharded head
would need a different CB strategy or a multi-pass N split, which is a kernel project, not a
layout switch.

Across the MLP (88 MB of weights per layer, 335 us) and the head, the decode matmuls run at
roughly 300-360 GB/s effective. That is the current small-M matmul ceiling in this runtime,
not a single op's misconfiguration.

## What this implies for the tuning change

- The collective shape has little padding to remove: the payload is a single tile row
  (`[1,1,B,2560]`, 5 KiB at width 1, 80 KiB at width 16) and the cost is almost all fixed
  startup and two-device handshake.
- A collective candidate can therefore win at most a fraction of 3 ms unless it also
  removes a launch or the matmul's DRAM round-trip.
- The step's dominant term is MLP weight streaming (24.9 ms of 48.1 ms) with weights
  already at bfloat4_b/bfloat8_b; the next material lever is a precision decision
  (`QWEN36_MLP_DOWN_BF4`, already screened and rejected on NLL) or a fused/faster
  projection kernel, not collective shaping.

## Non-claims

Diagnostic only. No admission, performance, or request-time claim. The traced numbers are
one session each, the component probe is width 1 only, and the width-16 mixer includes the
instance loop. These numbers do not establish a variant as better; they bound what the
collective work can recover.
