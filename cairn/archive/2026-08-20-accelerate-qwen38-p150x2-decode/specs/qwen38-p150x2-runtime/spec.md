# Qwen3.8-27B P150x2 Runtime Specification Delta

## ADDED Requirements

### Requirement: Decode trace replay is bounded and fail-closed
r[tenstorrent.qwen38_p150x2.traced_decode] The service MUST capture only an eligible static decode step. It MUST bind the trace to immutable runtime, model, slot, buffer, shape, dtype, and generation identities. It MUST restore recurrent state after capture. It MUST retain eager decode for every unsupported or invalid trace case.

#### Scenario: Eligible decode uses trace replay
r[tenstorrent.qwen38_p150x2.traced_decode.valid]
- GIVEN loaded weights, warmed programs, stable request buffers, and restored recurrent state
- WHEN an eligible greedy decode step executes
- THEN the service MUST refresh all mutable trace inputs before replay
- AND the replay MUST consume the prior request state exactly once.

#### Scenario: Invalid trace binding falls back
r[tenstorrent.qwen38_p150x2.traced_decode.fallback]
- GIVEN a missing, stale, mismatched, oversized, or unsupported trace binding
- WHEN decode admission validates the binding
- THEN the service MUST select eager decode before token execution
- AND the receipt MUST name the bounded fallback reason.

#### Scenario: Trace capture changes recurrent state
r[tenstorrent.qwen38_p150x2.traced_decode.state_drift]
- GIVEN trace capture output whose recurrent state differs from the pre-capture snapshot
- WHEN post-capture validation runs before the first request
- THEN validation MUST restore the snapshot and verify its identity
- AND startup MUST fail if restoration cannot be proved.

### Requirement: Decode buffers have explicit request ownership
r[tenstorrent.qwen38_p150x2.persistent_decode_buffers] The service MUST allocate a bounded set of stable decode slots. One admitted request MUST own exactly one slot. Each slot MUST own its token, position, RoPE, page-table, KV, GDN, output, and trace-binding resources. Slot reuse MUST follow a complete reset. Cleanup failure MUST quarantine the slot.

#### Scenario: Stable buffers serve one request
r[tenstorrent.qwen38_p150x2.persistent_decode_buffers.valid]
- GIVEN an empty slot and one admitted request
- WHEN the request executes two or more decode steps
- THEN every step MUST use the same slot generation and stable device resources
- AND no other live request MAY use those resources.

#### Scenario: Slot cleanup fails
r[tenstorrent.qwen38_p150x2.persistent_decode_buffers.cleanup_failure]
- GIVEN a completed, cancelled, timed-out, or failed request whose slot cannot pass reset validation
- WHEN the service releases request ownership
- THEN the slot MUST enter quarantine
- AND the service MUST reject reuse until a complete reset succeeds.

#### Scenario: Invalid slot transition is requested
r[tenstorrent.qwen38_p150x2.persistent_decode_buffers.invalid_transition]
- GIVEN a slot transition outside the typed lifecycle
- WHEN the pure lifecycle core validates the transition
- THEN validation MUST fail before device or filesystem effects occur.

### Requirement: Greedy selection transfers bounded shard winners
r[tenstorrent.qwen38_p150x2.shard_local_argmax] Each TP2 vocabulary shard MUST produce one finite maximum and one global token index. The host MUST select only from validated shard winners. Ties MUST select the lowest global token index. Admitted traced greedy decode MUST NOT transfer the complete vocabulary logits to the host.

#### Scenario: Valid shard winners select one token
r[tenstorrent.qwen38_p150x2.shard_local_argmax.valid]
- GIVEN complete non-overlapping shard ranges and one finite winner from each shard
- WHEN the pure reducer selects the global winner
- THEN it MUST return the token with the greatest logit
- AND equal logits MUST select the lowest global token index.

#### Scenario: Malformed shard result is rejected
r[tenstorrent.qwen38_p150x2.shard_local_argmax.invalid]
- GIVEN a missing winner, duplicate range, overlapping range, out-of-range index, or non-finite maximum
- WHEN shard-result validation runs
- THEN validation MUST fail before generated text is published.

### Requirement: Decode speedup claims require matched physical evidence
r[tenstorrent.qwen38_p150x2.decode_performance_admission] A traced candidate MUST match the eager path's greedy generated tokens for the profile corpus. A speedup claim MUST use order-balanced physical A/B evidence. The evidence MUST bind both paths to the same closure, checkpoint, tokenizer, workload, mesh, and generation settings. Nickel MUST own every admission bound.

#### Scenario: Physical candidate passes
r[tenstorrent.qwen38_p150x2.decode_performance_admission.valid]
- GIVEN complete physical A/B evidence with exact token parity and clean slot reset
- WHEN the pure admission core evaluates decode rate, TTFT, and tail latency
- THEN every profile-owned bound MUST pass
- AND the receipt MAY admit traced decode for the bound closure and workload.

#### Scenario: Candidate lacks complete evidence
r[tenstorrent.qwen38_p150x2.decode_performance_admission.invalid]
- GIVEN missing parity, cleanup, machine, attribution, latency, throughput, or identity evidence
- WHEN admission evaluates the candidate
- THEN it MUST withhold the speedup claim
- AND it MUST retain eager decode as the admitted fallback.
