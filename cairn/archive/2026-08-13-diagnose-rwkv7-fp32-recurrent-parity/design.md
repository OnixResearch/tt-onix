## Context

The package-default BF16 model-matmul contract causes the known token divergence. A diagnostic HiFi4 and FP32-destination-accumulation contract restores the pinned sixteen-token continuation.

The corrected device FP32 recurrent-matrix BLAKE3 is `9aca7260763e0a728b30175c40f65d43eee5743b68aea593675f94936d507c98`. The pinned CPU authority is `0d904c3766a7cb13aaaeb13266e65623d516056ab6cfa2d697ccb3610dc593d7`.

## Goal and completion evidence

The goal is to identify the first exact recurrent-matrix divergence and its causal numerical mechanism.

Completion requires one aggregate receipt with the model step, layer, recurrence operation, data type, element count, first differing element, finite maximum error, and deterministic CPU and device identities. A correction is valid only if all sixteen tokens and the final FP32 recurrent matrix match exactly.

False completion includes token-only agreement, final-hash-only comparison, an unmatched CPU or device harness, raw-tensor retention, an unbounded device loop, or candidate-v3 creation before exact parity.

## Serial portfolio registry

Subagent consent is absent. The following passes are serial and correlated.

1. **Authority-shape lens**: prove that CPU and device hash the same normalized `[layer, head, value, key]` logical matrix bytes.
2. **Input-materialization lens**: compare the BF16-derived recurrence inputs before each FP32 recurrence update.
3. **FP32-operation lens**: compare `state_dot_a`, `rank_update`, `decay`, `outer_update`, `first_add`, and `next_state`.
4. **Update-order lens**: test whether operation grouping or matmul accumulation order causes the first exact mismatch.

A route is falsified when its complete input and output identities match before the first known divergence. A route remains blocked when its missing check is equivalent to full parity.

## Decisions

### Decision: Localize before changing arithmetic

**Choice:** Add deterministic per-step and per-layer matrix identities before any correction.

**Rationale:** The final hash does not show whether divergence starts at the first prompt token or accumulates later.

### Decision: Preserve matched execution harnesses

**Choice:** Compare the pinned CPU authority with the same high-precision, generic-WKV, normal-dispatch device path that restored all tokens.

**Rationale:** An alternate fused path, BF16-state harness, or package-default matmul path cannot establish this parity contract.

### Decision: Retain aggregate evidence only

**Choice:** Retain hashes, counts, the first differing index, and finite maximum errors. Remove raw tensors, caches, compiler output, profiler output, and operational logs.

**Rationale:** The diagnosis needs numerical boundaries, not model payload retention.

## Evidence budget

- Retrieval: the archived blocker, CPU authority, runtime recurrence, and package-matched TT-NN matmul contract.
- Device-free: one focused baseline and one focused final set.
- Physical: one device, normal dispatch, at most sixteen model steps per discriminator, and no more than three discriminators.
- Simulator: only if new device execution code is introduced and package-matched operations do not already cover it.
- Retention: aggregate JSON and Markdown only.

## Result

The first exact divergence is model step `0`, layer `0`, operation `outer_update`. The initial state, `state_dot_a`, `rank_update`, and `decay` match exactly.

A CPU outer product from the exact BF16-derived device key and value rows matches the device outer product exactly. This proves that recurrence-input materialization causes the first mismatch. It falsifies FP32 recurrence arithmetic as the cause of that baseline mismatch.

The bounded correction keeps the embedding output, prior attention vector, and layer-zero weights in FP32. It fails the initial boundary. Its key and value rows differ from the pinned CPU authority. Its FP32 outer product also differs from a CPU outer product that uses the exact corrected device inputs.

The required initial boundary did not match. Therefore, the sixteen-token correction run did not start. Candidate v3 remains absent, and production remains `[2,4,8]`.

## Stop conditions

Stop with `validated` when exact token and matrix parity pass. Stop with `blocked` when the first causal boundary is proven but its correction fails exact parity. Stop with `exhausted` after three physical discriminators or when all four mechanisms are falsified or blocked.

This diagnosis stops with `blocked`.

## Risks / Trade-offs

- Reading each bounded matrix boundary adds synchronization and can change timing. Normal dispatch and deterministic identities limit the claim to arithmetic parity.
- Exact FP32 parity is stricter than token parity. This strictness is required for admission.
- CPU and TT-NN matmul implementations can use different valid accumulation orders. If so, exact parity can require a package-matched operation-level correction rather than a tolerance.
