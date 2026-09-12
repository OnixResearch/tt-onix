## Context

The existing B=8 profile, contracts, policy, and binding already exist in the
package. The rejected receipt is
`pkgs/qwen36/evidence/2026-08-26/qwen38-27b-batch-admission-b8-receipt.json`
with BLAKE3 `e6641e7efa02f384e550e0621fea9ea97f1eaffe5d258fb3634312cc183a3fa5`.
The binding carries `physically_admitted = false`.

The measured margins are small. Aggregate speedup missed by 0.1185 of a ratio,
which is about two percent of the bound. Per-user retention missed by 0.0148.
TTFT missed by 0.0792 of a ratio.

A small margin is not evidence that the candidate passes. It is evidence that
the candidate is close enough that a prefill-scheduling change can plausibly
decide it. The prefill change exists for an independent reason and is being
measured on its own terms. This change consumes it.

## Decisions

### Decision: Reuse the rejected profile without editing a bound

**Choice:** Run the existing B=8 profile, policy, and contracts unchanged. Take
no edit to any threshold, corpus size, root count, token count, or order.

**Rationale:** The archived rejection recorded that "the bound, however, was
predeclared and the rejection stands unchanged." Editing a bound to convert a
known result would reproduce the failure mode this repository exists to
prevent. The change edits the runtime, not the gate.

### Decision: Treat this as a fresh measurement, not a retry

**Choice:** Record a new receipt with a new package identity, a new runtime
identity, and a new prefill mechanism identity. Keep the old receipt and its
binding history in place.

**Rationale:** The mechanism changed, so the old receipt does not describe the
new execution. A fresh balanced run keeps both results comparable and leaves the
rejection as valid history.

### Decision: Require both margins and correctness, with no partial admission

**Choice:** Admit B=8 only when every bound passes. A pass on aggregate speedup
with a failed per-user retention bound is a rejection.

**Rationale:** The profile's purpose is served quality at width, not raw
aggregate. The B=4 admission used the same combined contract and the same
retention floor.

### Decision: Keep B=32 out of scope

**Choice:** Do not extend the profile, and do not attempt B=32 while fixing
this.

**Rationale:** B=32 fails on `per_core_M % out_subblock_h == 0` in the matmul
program config at TP=2. That is a divisibility defect with a different owner,
and it is not a bound question.

## Risks / Trade-offs

- The prefill mechanism may not change the outcome. The expected result is
  another rejection, and that is a valid deliverable.
- A second rejection closes the prefill explanation for this near miss. The
  follow-up would then be decode-step scaling at width, not prefill.
- The batched-prefill path may shift TTFT in either direction. The profile
  bounds TTFT at most 9.0, so an improvement and a regression are both visible.
