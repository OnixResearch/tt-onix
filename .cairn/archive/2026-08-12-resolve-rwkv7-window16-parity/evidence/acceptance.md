# Window-16 parity diagnosis result

## Result

Window `16` remains blocked. Production still admits only `[2, 4, 8]`.

The bounded generic TT-NN diagnostic matched the pinned CPU token through generated index `10`. The rejected fused path first differed at index `10`:

- CPU and generic TT-NN: `46170`
- rejected fused WKV: `64390`
- prior matching index: `9`

The source audit found a precision-boundary mismatch. The generic path packs the FP32 WKV readout to BF16 before group normalization. The rejected fused path keeps that readout in FP32.

## Added checks

The evidence crate now has a pure first-divergence comparator. Positive tests accept equal boundaries. Negative tests report index `10` and reject malformed or non-finite boundaries.

The fused reference now models the generic BF16 readout boundary. Its negative regression keeps the rejected FP32 behavior and proves that asymmetric values distinguish the two paths.

The kernel source guard now requires:

- the diagnostic BF16 readout branch;
- the unchanged production FP32 readout branch;
- the explicit BF16 readout pack;
- positive and negative source tests for those boundaries.

## Physical blocker

A diagnostic-only runtime enabled the BF16 readout boundary. Production and the rejected candidate kept the build flag disabled.

The asymmetric physical probe used device `0`. It compared FP32 recurrent state and BF16 corrected attention rows. The first attempt reached its 30-minute cold-compilation bound. The bounded retry reached its 60-minute bound after device initialization and program-cache enablement.

Neither attempt wrote a kernel-execution receipt. Therefore, this change does not claim physical parity or a confirmed correction.

The aggregate blocker is in `physical-diagnostic-blocker.json`. Raw compiler and watcher paths remain outside the repository.

## Decision

Do not create a new immutable window-16 admission candidate. Do not start simulator, profiling, A/B, canary, monitoring, or deployment gates.

A later change can resume with the diagnostic package after the fused kernel compiles within a named bound. It must first pass exact asymmetric state and output parity, then exact full-model token and FP32-state parity.
