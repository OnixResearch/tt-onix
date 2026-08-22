## Context

The current target emits about one token per traced replay. A useful MTP verifier must reuse target-model work across several teacher-forced tokens while preserving hybrid-model state. Qwen3.8-27B has recurrent GDN state, convolution history, and paged full-attention KV state. A timing result without state equivalence cannot establish feasibility.

The packaged TT-Metal 0.74 runtime does not expose the newer fused GDN chunk operation. The adopted model still has a bounded masked-prefill path that can start from carried state. The probe can use that path to identify whether the pinned runtime has a correct and economically useful target-verification base.

## Decision

Keep policy and evidence evaluation in a pure `mtp_probe_core` component. Put TTNN calls, state snapshots, timing, and receipt I/O in diagnostic adapters and the command shell.

The operator must select the probe explicitly. The shell will:

1. Open the exact P150x2 mesh and load the pinned model.
2. Warm all width-specific programs before it parks a decode trace.
3. Run at least five complete samples for widths 1, 2, and 4.
4. Recreate the same prompt state before sequential and teacher-forced paths.
5. Release the parked trace before verifier allocations, then recapture for the next sequential sample.
6. Compare every target token and the final GDN recurrent and convolution states.
7. Record device execution and host readback separately.
8. Emit an immutable diagnostic receipt and restore device ownership.

The pure core will validate finite positive durations, expected widths, sample counts, identities, token parity, state bounds, and execution order. It will report target-path amortization without selecting MTP. MTP selection still requires a measured drafter, local acceptance, rollback behavior, and matched end-to-end evidence.

The architecture correction uses the pinned checkpoint configuration and the observed 64-layer cache pattern. Full-attention layers occur every fourth layer, which gives 16 full-attention and 48 GDN layers.

## Alternatives

A GDN-only fusion was rejected as the first implementation because no local component timing establishes it as dominant. Static cache sizes show that common MLP weights form the larger decode byte bucket.

A complete MTP implementation was rejected for this change because the target verification economics and hybrid-state semantics are not yet measured on P150x2.

A prefill-only width test was rejected because it starts from zero state and does not exercise verification after an existing prompt.

## Safety and Evidence Boundaries

The probe is diagnostic-only and cannot satisfy decode admission. Production wrappers remain eager. A failed, missing, unstable, or numerically mismatched probe selects no optimization.

No result may claim MTP acceptance, speculative speedup, sampling correctness, or production readiness. Those claims require the unused checkpoint MTP weights and a later physical change.

## Non-Goals

- No MTP weight loading or drafter implementation.
- No production decode-mode change.
- No continuous batching.
- No runtime pin update.
- No GDN kernel fusion.
