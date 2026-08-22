## Context

The exact width-four native verifier amortizes target work by `1.9469233481156285x`, and the MTP pilot accepted `12` of `15` drafts. The latest selector control removed full-vocabulary readback by construction, but MTP draft compute remained unstable and complete economics failed. The package-matched `TPAttention` path uses fixed-shape batch-one decode, explicit position tensors, and internal K/V tensors. A trace can therefore reuse one compiled draft step if all mutable inputs and cache addresses remain stable.

The internal MTP cache is append-addressed by the supplied position. This change proves only that trace warmup, capture, and replay preserve the bounded prefix for the same inputs. Rejected-row overwrite remains a later hypothesis. This change does not claim multi-round MTP rollback.

## Decisions

### Decision: Trace one MTP decode step as a narrow adapter

**Choice:** Add a `TracedMtpDraftAdapter` that owns stable token, position, RoPE, feedback, selector, and output tensors. Capture includes embedding, fusion, the MTP decoder layer, final normalization, local LM-head projection, and shard-local top-one reduction.

**Rationale:** This boundary contains the unstable fixed-shape draft work without changing the target model, MTP weights, or greedy selection policy. It also keeps trace ownership and cleanup outside the deterministic evaluator.

### Decision: Compare complete roots, not isolated replay microbenchmarks

**Choice:** Run eager and traced roots in balanced `eager,traced,traced,eager` blocks. Retain at least five samples per route and prompt. Count MTP prefill, draft replay, selection, target verification, target readback, target rollback, and any MTP rollback cost in complete economics.

**Rationale:** The previous selector result showed that a faster local operation can still lose end to end. Complete emitted-token economics remains authoritative.

### Decision: Use explicit trace bindings and fail closed

**Choice:** Record device buffer addresses for every mutable trace input, MTP K/V tensor, and persistent output. Reject replay after any identity, shape, dtype, or address change. Release the trace before cache reallocation or route changes.

**Rationale:** TTNN traces bind device addresses. Python object identity is not sufficient evidence.

### Decision: Bound cache evidence to capture idempotence

**Choice:** Record the K/V row at the captured position after warmup and after capture. Require exact equality for the same token, feedback, and position. Do not clone or copy the full maximum-length cache.

**Rationale:** This check proves that trace setup does not corrupt the measured prefix. A later multi-round change must separately prove rejected-row overwrite with a committed token.

## Risks / Trade-offs

- Trace capture can exceed the one-gigabyte region or include an unsupported operation. The adapter must release partial traces and report the exact blocker.
- Capturing the draft step writes MTP K/V state. Measurement can proceed only after exact restoration or an exact same-position idempotence check.
- Stable replay can still fail complete economics because MTP prefill, target verification, and rejection costs remain.
- Physical execution requires an authorized outage of `qwen38-p150x2.service`. Device-free checks cannot establish trace replay or silicon timing.
- This diagnostic work does not change production wrappers, which continue to discard `mtp.*` weights.
