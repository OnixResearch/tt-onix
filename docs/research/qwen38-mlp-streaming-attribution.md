# Qwen3.8-27B decode: MLP streaming attribution

Evidence-only arithmetic; no new measurement. Scope is the TP2 `step` route on two
P150a devices, not a service benchmark ([A], opening and “What was measured”).

## Step budget and timing boundaries

The reported measured step is **44–50 ms**; the **48.14 ms component sum** is a
sum of separately timed component medians weighted by layer count, **not an
end-to-end step measurement** ([A], “Component medians”).

| Width-1 component | Median per call, ms | Multiplicity | Weighted ms |
| --- | ---: | ---: | ---: |
| GDN mixer | 0.3159 | 48 | 15.16 |
| MLP | 0.3894 | 64 | 24.92 |
| Full-attention mixer | 0.3705 | 16 | 5.93 |
| LM head + argmax | 2.0842 | 1 | 2.08 |
| Embedding | 0.0497 | 1 | 0.05 |
| Component sum (not end-to-end) | | | **48.14** |

All table inputs and reported rounded totals: [A], “Component medians”. Arithmetic
from the displayed medians gives 48.1467 ms; the source's 48.14 ms need not equal
the sum recomputed from rounded inputs. MLP share is approximately
`24.92 / 48.14 = 51.8%`; the remainder is approximately 23.22 ms (arithmetic, [A]).

The separate trace harness replays each captured operation group **200 times per
sample**, removing host dispatch ([A], “What was measured”). Its MLP forward,
excluding norm/add, is **335.3 µs at both widths 1 and 16**; multiplying that traced
group by 64 gives **21.4592 ms, a sum of traced groups, not an end-to-end step**
([A], “Traced groups”; multiplication is arithmetic). The component MLP median is
389.4 µs, leaving `64 × (389.4 − 335.3) µs = 3.4624 ms` between instruments;
this is **not an isolated measurement of norm/add or kernel overhead** ([A]).
Thus “24.9 ms weight streaming” in [B] is a broad MLP attribution, not a measured
sum of projection-only kernel times. No gate/up/down timing split is recorded.
Only the traced forward is shown flat with width; there is no width-16 component
budget or proof that the complete step is flat ([A], “Non-claims”).

For contrast, GDN mixer trace groups are **265.1 / 389.7 µs** at widths **1 / 16**
([A], “Traced groups”). The separate staged-recurrence experiment records
**51.2–51.5 / 186.5–186.7 µs** for recurrence and **70.4–70.7 / 201.3–201.5 µs**
including `conv_prep`; these are traced subgroups, not whole mixers or steps
([C], “Method”, “Scalar-read staging”). Its **49.9 / 44.4 ms** served/staged
end-to-end medians are uncontrolled single sessions, not a timing comparison ([C]).

Older-route budgets must not be merged: [D], “A1” has **54.34 ms full trace replay**
versus `26.34 + 23.86 + 5.65 + 2.07 = 57.92 ms`, a component sum, not end-to-end.
[E], “The width-dependent cost” has no current width-16 MLP split. [F], `decision`,
rejects its historical attribution and separates measured from modeled device time.

The shell probe starts fresh servers, selects `QWEN35_GDN_DECODE_FUSION=full`,
pools `token_latency_seconds`, drops first/last samples when length exceeds three,
and prints median/mean: not isolated MLP or HTTP wall time ([G], lines 49–56,
77–87, 148–173). It is a screen, not admission ([G], lines 4–25).
No `t2-*` / `t3-*` splits exist here; [B], “Verdict” leaves those tasks unexecuted.

## Weight-byte model and the hardware ceiling

All byte rates below are **per device**, decimal GB/s. Model inputs are hidden
**5120**, intermediate **17408**, TP **2** ([H], “Carried P150x2”); gate/up are
**bfloat4_b**, down **bfloat8_b** ([A], “Traced groups”). The TP model halves each
matrix: gate/up column shards and down row shards (also documented in [I],
`load_mlp_weights`). Arithmetic: each full matrix has `5120 × 17408 = 89,128,960`
parameters; each device has **44,564,480**. Three matrices give **267,386,880**
parameters per layer, **17,112,760,320** across 64 layers ([A], [H]; arithmetic).

| Per-device matrix | Logical shape | Payload bytes | Packed tile bytes |
| --- | --- | ---: | ---: |
| Gate | 5120 × 8704 | 22,282,240 | 25,067,520 |
| Up | 5120 × 8704 | 22,282,240 | 25,067,520 |
| Down | 8704 × 5120 | 44,564,480 | 47,349,760 |
| Per-layer total | | **89,128,960** | **97,484,800** |

Table is arithmetic from [A], [H], [I]. Payload counts use 0.5 / 1 byte per
BF4 / BF8 datum; packed counts use **576 / 1088 bytes per 32×32 tile**, including
**64 exponent bytes**, from [J], `BFLOAT4_B_TILE_HW` / `BFLOAT8_B_TILE_HW`.
This is a one-read weight model, not measured DRAM traffic; it excludes activation,
collective, padding/layout effects beyond those tiles, and rereads.

At the **335.3 µs** traced MLP-forward time ([A]), arithmetic gives:

| Model | Bytes / seconds | Effective rate |
| --- | --- | ---: |
| TP2 exponent-exclusive payload | 89,128,960 / 0.0003353 | **265.82 GB/s** |
| BFP exponent-inclusive tiles | 97,484,800 / 0.0003353 | **290.74 GB/s** |
| Existing [A] range | 88,000,000 / 0.000335 = 262.69 GB/s, not its stated range | **~300–360 GB/s stated** |

The **88 MB** figure originates in [A], “LM head”; no underlying byte-count
receipt is attached there. It approximates exponent-exclusive payload, not tiled
storage. The range is **not reproducible from its own MLP bytes/time pair**.
The TP2 collective payload **[1,1,B,2560]**, **5 / 80 KiB** at widths **1 / 16**
([A], “What this implies”) is `B × 2560 × 2` bytes: logical BF16 activation
payload, not MLP weights and not exponent-inclusive BFP storage. The reference
weight loader calls `shard_w`, which sets `layout=ttnn.TILE_LAYOUT` ([I], [L]);
[J] includes exponents. **290.74 GB/s is the consistent tiled-weight effective
model** for this forward, whereas **265.82 GB/s is payload-only**, not physical
weight bytes/s. This reconstructs bytes from source, not a measured DRAM counter
or isolated matmul time; [I]/[L] are upstream references, not the captured harness.

The exact preferred reference is **512 GB/s per P150a** in
`.pi/references/tenstorrent/docs-site/core/aibs/blackhole/index.md`,
“Card Comparison Table”, “Memory Bandwidth” row ([K]). Arithmetic:
`265.82 / 512 = 51.9%`, `290.74 / 512 = 56.8%`.
The term is bandwidth-bound at **just over half the rated ceiling**, not near
hardware saturation; [A]'s runtime “ceiling” is not the physical ceiling.
This leaves conceptual room to investigate residency/fusion, **not evidence
that either will pay off**, and does not prove any variant exists.

## LM head: separate bytes, separate time

[A], “LM head” reports a per-device **[5120,124160] bfloat8_b** weight and
**1784.7 µs** traced matmul at width 1: arithmetic gives **635,699,200 payload
bytes / 0.0017847 s = 356.2 GB/s**, or **69.6%** of [K]'s bandwidth. With [J]'s
exponents, **675,430,400 bytes** imply **378.5 GB/s**, **73.9%** of [K]. The
**2.0842 ms** head+argmax component is **4.33%** of the **48.14 ms component sum,
not an end-to-end share** ([A]; arithmetic); it already contains the matmul.
Do not add the head bandwidth to the MLP bandwidth: they describe different time
intervals. A common-denominator illustration is packed MLP **6.2390 GB** plus
head **0.6754 GB** per device per step, giving respectively **129.6 + 14.0 =
143.6 GB/s** over the **48.14 ms component sum, not an end-to-end measurement**
(arithmetic from [A], [H], [J]). These are disjoint modeled streams, not a claim
about average total device traffic.

## Sensitivity — arithmetic only

Using the reproducible rates above, a hypothetical **1.2×** rate means
**319.0 / 348.9 GB/s** payload/tiled; **2×** means **531.6 / 581.5 GB/s**
(arithmetic from [A], [J]). The latter exceeds [K]'s **512 GB/s** if interpreted
as unchanged bytes streamed from DRAM; it would require a changed traffic model
or is infeasible under that interpretation. Apply the factor to the entire
**24.9 ms** attribution ([B], “Dominant term”), holding everything else fixed:
this is an optimistic arithmetic assumption, not an isolated-kernel measurement.

| Hypothesis | Remaining term | Recovered time |
| --- | ---: | ---: |
| 20% faster throughput (1.2×) | 24.9 / 1.2 = 20.75 ms | **4.15 ms** |
| 2× throughput | 24.9 / 2 = 12.45 ms | **12.45 ms** |

If “20% faster” instead means 20% less time, arithmetic gives **4.98 ms** recovered.
No performance or service-latency claim; no claim any faster variant exists,
passes retention, or is achievable. No hardware/simulator was run for this note.

## Repository sources

[A]: ../../.cairn/changes/qwen38-tp2-collective-shape-tuning/evidence/step-attribution-2026-10-21.md
[B]: ../../.cairn/changes/qwen38-tp2-collective-shape-tuning/evidence/ceiling-verdict-2026-10-22.md
[C]: ../../.cairn/changes/optimize-gdn-decode-kernel-grid-and-prep/evidence/step-kernel-scalar-staging.md
[D]: ../../.cairn/changes/optimize-gdn-decode-kernel-grid-and-prep/evidence/a1-a4-attribution.md
[E]: ../../.cairn/changes/optimize-gdn-decode-kernel-grid-and-prep/evidence/width-dependent-cost.md
[F]: ../../pkgs/qwen36/evidence/2026-08-22/qwen38-27b-decode-component-attribution-receipt.json
[G]: ../../scripts/qwen38-step-probe.sh
[H]: ../qwen38-27b-adoption.md
[I]: ../../.pi/references/tenstorrent/tt-metal-latest/models/demos/blackhole/qwen36/tt/mlp.py
[J]: ../../.pi/references/tenstorrent/tt-metal-docs/tt_metal/api/tt-metalium/constants.hpp
[K]: ../../.pi/references/tenstorrent/docs-site/core/aibs/blackhole/index.md
[L]: ../../.pi/references/tenstorrent/tt-metal-latest/models/demos/blackhole/qwen36/tt/tp_common.py
