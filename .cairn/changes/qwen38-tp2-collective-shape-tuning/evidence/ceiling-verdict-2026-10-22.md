# Collective ceiling verdict, 2026-10-22

This is a diagnostic verdict from the existing `evidence/step-attribution-2026-10-21.md`, read against this change's proposal and design. No new measurement, hardware execution, or candidate implementation was performed for this verdict.

## Measured collective ceiling

A decode step issues **112 row-parallel collectives**: 64 MLP down-projection reductions and 48 GDN out-projection reductions. The traced cost of **25.1–30.0 µs** per collective gives **2.8–3.4 ms** against a measured **44–50 ms** step, approximately **6–7 %**. This is the measured collective-work envelope, not a promised recoverable gain.

The payload is one tile row, `[1,1,B,2560]`: **5 KiB at width 1** and **80 KiB at width 16**. The attribution records little padding to remove and mostly fixed startup and two-device handshake costs.

## Already-rejected mechanisms

`matmul_reduce_scatter_decode` already exists in `tp_common.py`. Its own docstring records its rejection for decode **M=1**: the 2-D matmul collapses to roughly eight cores and the fused form loses. Separate matmul plus `reduce_scatter_minimal_async` is the already-selected mechanism, not an untried alternative.

The DRAM-sharded LM-head recipe is rejected on a circular-buffer bound: **9,046,592 B** of static circular buffers requested against **1,572,864 B** of L1. It fails before timing. A different CB strategy or multi-pass N split would be a kernel project, not a layout transition.

## Dominant term and remaining lever

MLP weight streaming contributes **24.9 ms of the 48.1 ms** width-1 component sum. The evidence places effective decode matmul bandwidth across the MLP and LM head at approximately **300–360 GB/s**; the MLP forward is flat between widths 1 and 16. This is the dominant term, not collective shaping.

`QWEN36_MLP_DOWN_BF4` was already screened and rejected on the NLL floor. That rejection stands. The next material lever named by the evidence is a **fused/faster projection kernel**, which is a different change rather than another collective-shape candidate.

## Verdict: re-scope rather than continue t2–t6

The design's bounded candidate set—a TP2 grid, a layout transition, or a small-message collective—can recover at most a fraction of approximately **3 ms** within the measured collective envelope. The evidence does not justify continuing t2–t6 as written as a route to a material step improvement. Those tasks remain open and are not being executed as written.

Recommend re-scoping to the projection/MLP path as a successor change, with a separately defined kernel mechanism and numerical-retention screen. This verdict does not silently expand the current candidate set or admit a replacement.

## Non-claims

- Diagnostic evidence only: each traced number is from one session, and the component probe is width 1 only.
- No admission, performance, or request-latency claim; synchronized component and replay timings do not establish service latency or a surviving variant.
- No claim that a fused/faster projection kernel is achievable within the retention floor. The MLP-down BF4 rejection is evidence against assuming that a precision-based shortcut will satisfy it.
- No new hardware run, simulator run, or measurement is reported here.

## Appended correction, 2026-10-22: reproducible MLP bandwidth

This appends a numeric correction to the verdict above; it does not re-judge or
change its qualitative conclusion: the MLP is bandwidth-bound, not collective-bound.
No new measurement was performed.

The quoted **300–360 GB/s** range is not reproducible from the MLP bytes/time pair
in `step-attribution-2026-10-21.md`, “LM head”: **88 MB / 335 µs = 262.7 GB/s**.
At its measured **335.3 µs** forward time (“Traced groups”), the shipped
gate/up **bfloat4_b** and down **bfloat8_b** give **265.8 GB/s** with the TP2
exponent-exclusive payload model (**89,128,960 B / 0.0003353 s**) or **290.7 GB/s**
with exponent-inclusive tiled storage (**97,484,800 B / 0.0003353 s**).
The tiled figure is the appropriate effective weight-byte model for tiled BFP
weights, not a measured DRAM counter. The original 88 MB is approximate payload,
not the tiled allocation. The forward time is not a projection-only timing.

The exact packaged reference ceiling is **512 GB/s per P150a** in
`.pi/references/tenstorrent/docs-site/core/aibs/blackhole/index.md`, “Card Comparison
Table”, “Memory Bandwidth” row. Arithmetic gives **51.9–56.8%** utilization:
bandwidth-bound at just over half the rated ceiling, not hardware saturation.
This does not establish that residency/fusion work will pay off or that a variant
exists. The byte derivation, shipped precision sources, LM-head separation and
arithmetic-only sensitivities are in
`docs/research/qwen38-mlp-streaming-attribution.md`, “Weight-byte model and the
hardware ceiling” through “Sensitivity”; tile byte constants are in
`.pi/references/tenstorrent/tt-metal-docs/tt_metal/api/tt-metalium/constants.hpp`,
`BFLOAT4_B_TILE_HW` / `BFLOAT8_B_TILE_HW`.
