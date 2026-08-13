## Context

Candidate v2 has exact fused-boundary parity for the prior diagnostic fixture. It still fails full-model CPU token parity at index `10`.

The normal and persistent device paths produce identical tokens and final device state. This result excludes Metal Trace replay as the first cause.

## Goal and completion evidence

The goal is to find and correct the first full-model divergence. Completion requires one bounded diagnostic receipt that identifies an operation boundary and its causal mechanism.

A validated correction also requires exact pinned CPU token and FP32 recurrent-state parity for all sixteen generated tokens. Only then can a new immutable candidate start admission gates.

False completion includes device-only agreement, token-only agreement without CPU state parity, an unbounded search, raw-log retention, or reuse of v2 identity.

## Search registry

The diagnosis uses three serial and correlated lenses:

1. **Fused-selection lens**: compare generic TT-NN and fused WKV at every layer for the known failing prefix.
2. **Non-WKV lens**: compare layer input, attention output, channel-mix output, final normalization, and logits.
3. **Accumulation lens**: compare CPU and device recurrent state before the first token mismatch.

The smallest discriminating check runs first. A falsified lens does not cause a wider hardware run.

## Decisions

### Decision: Use one bounded normal-dispatch diagnosis

**Choice:** Run one device on the shortest prefix that reaches generated-token index `10`. Do not use persistent trace.

**Rationale:** Candidate v2 already proves normal and persistent agreement. Normal dispatch removes trace setup and replay from the diagnosis.

### Decision: Hash complete boundaries and retain aggregate differences

**Choice:** Compute deterministic boundary hashes and aggregate error values. Keep raw tensors and logs outside the repository.

**Rationale:** Boundary hashes localize exact agreement. Aggregate errors classify inexact agreement without retaining model data.

### Decision: Create candidate v3 only after root-cause proof

**Choice:** Keep diagnosis code separate from admission profiles. Bind a later v3 profile to the final diagnostic receipt.

**Rationale:** Candidate v2 is immutable and rejected. A new identity is valid only after the correction is known.

## Evidence budget

- Device-free checks: one complete focused set before and after code changes.
- Simulator: one single-card smoke only if diagnostic code changes device execution.
- Physical diagnosis: one device, one normal-dispatch request, and at most eleven generated tokens.
- Correction probe: one device and sixteen generated tokens only after the first divergence is proven.
- Retention: aggregate JSON and Markdown only.

## Risks / Trade-offs

- Hashes can show the first exact divergence but not its numerical direction. Aggregate error values provide the required direction.
- A token mismatch can result from an earlier small error. The diagnosis must compare recurrent state and logits before argmax.
- Physical devices can be busy. The shell must fail closed without resetting an owned device.
