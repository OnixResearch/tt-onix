# Qwen38 P150x2 Runtime Specification

## Purpose

Defines the `qwen38-p150x2-runtime` capability.

## Requirements

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

### Requirement: Continuous batching remains unsupported without an admitted execution base
r[tenstorrent.qwen38_p150x2.continuous_batching_non_support] The repository MUST keep Qwen3.8-27B P150x2 request execution serialized while its required traced execution base is unadmitted. It MUST NOT add or claim a continuous-batching scheduler, batched state path, batch profile, or aggregate throughput result from single-request evidence. Future continuous-batching support MUST use a new validated Cairn change with an admitted execution base and complete matched physical evidence.

#### Scenario: Traced execution base is unadmitted
r[tenstorrent.qwen38_p150x2.continuous_batching_non_support.valid]
- GIVEN the required traced execution base has a rejected physical receipt
- WHEN continuous-batching support is evaluated
- THEN the service MUST remain serialized
- AND no continuous-batching implementation or throughput claim MUST be added.

#### Scenario: A future traced path is admitted
r[tenstorrent.qwen38_p150x2.continuous_batching_non_support.invalid]
- GIVEN an execution base becomes admitted after this decision
- WHEN an operator requests continuous batching
- THEN this archived decision MUST NOT imply batch support
- AND a new validated change with batch-specific physical evidence MUST be required.

### Requirement: BF4 weight candidates remain experimental without complete admission evidence
r[tenstorrent.qwen38_p150x2.low_precision_non_admission] The repository MUST keep BFP8 as the production precision for the Qwen3.8 weight families covered by the existing opt-in BF4 gates until one named candidate has complete numerical and matched physical evidence. An environment flag, supported cache format, PCC result, static byte count, simulator result, or incomplete receipt MUST NOT imply production admission. Future admission MUST use a new validated Cairn change.

#### Scenario: Candidate evidence is incomplete
r[tenstorrent.qwen38_p150x2.low_precision_non_admission.valid]
- GIVEN a BF4 candidate lacks any required operator, logit, recurrent-state, token-parity, identity, or physical metric
- WHEN production precision admission is evaluated
- THEN the candidate MUST remain unadmitted
- AND BFP8 MUST remain the production profile.

#### Scenario: An experimental BF4 gate exists
r[tenstorrent.qwen38_p150x2.low_precision_non_admission.invalid]
- GIVEN an opt-in BF4 gate or cache format exists in the package
- WHEN an operator requests production admission
- THEN that implementation fact MUST NOT imply correctness, quality, or speed
- AND a new validated change with complete evidence MUST be required.

### Requirement: TP2 tuning selects no winner from incomplete sweep evidence
r[tenstorrent.qwen38_p150x2.tp2_tuning_non_admission] The repository MUST keep the current Qwen3.8 TP2 configurations and safe fallbacks when no complete attributed baseline profile and comparable legal sweep exist. Static shape analysis, another mesh size, compatibility behavior, simulator results, or isolated operation data MUST NOT imply a TP2-tuned configuration or model speedup. Future TP2 tuning admission MUST use a new validated Cairn change with profile-first and matched end-to-end physical evidence.

#### Scenario: TP2 sweep evidence is incomplete
r[tenstorrent.qwen38_p150x2.tp2_tuning_non_admission.valid]
- GIVEN baseline attribution or any required comparable candidate result is missing
- WHEN deterministic TP2 winner selection runs
- THEN it MUST select no winner
- AND current configurations and safe fallbacks MUST remain selected.

#### Scenario: An isolated operation result becomes available
r[tenstorrent.qwen38_p150x2.tp2_tuning_non_admission.invalid]
- GIVEN one operation result is faster without complete model evidence
- WHEN an operator requests TP2 tuning admission
- THEN that isolated result MUST NOT imply a model speedup or generated winner set
- AND a new validated change with matched end-to-end evidence MUST be required.

### Requirement: Traced decode uses request-bounded prefill capacity
r[tenstorrent.qwen38_p150x2.traced_decode_ttft] The traced-decode candidate MUST size each prefill page table from the bounded prompt and generation token counts. The prefill table MUST be a monotonic prefix of the stable decode table. The trace-bound decode table MUST retain its maximum profile capacity, shape, contents, and device address across requests. Invalid or capacity-exceeding requests MUST fail before prefill dispatch.

#### Scenario: Request uses a bounded prefill table
r[tenstorrent.qwen38_p150x2.traced_decode_ttft.valid]
- GIVEN positive prompt and generation counts within the trace capacity
- WHEN the pure block-count rule and adapter build the prefill table
- THEN its width MUST equal the rounded-up required block count
- AND its entries MUST equal the same prefix of the stable decode table.

#### Scenario: Request exceeds trace capacity
r[tenstorrent.qwen38_p150x2.traced_decode_ttft.capacity]
- GIVEN a prompt and generation bound that need more blocks than the trace capacity
- WHEN request block-count validation runs
- THEN it MUST reject the request before prefill dispatch
- AND it MUST NOT resize or replace the trace-bound decode table.

#### Scenario: Prefill width changes trace binding
r[tenstorrent.qwen38_p150x2.traced_decode_ttft.binding]
- GIVEN a request-bounded prefill table and an existing decode trace
- WHEN trace binding validation runs before replay
- THEN the stable decode page-table identity MUST remain unchanged
- AND any identity drift MUST select the fail-closed eager path.

### Requirement: TTFT admission uses phase-attributed matched evidence
r[tenstorrent.qwen38_p150x2.traced_decode_ttft_evidence] Each measured eager and candidate response MUST record finite non-negative reset, prefill-device, first-token-readback, trace-preparation, and total TTFT durations. It MUST record prefill and decode page-table widths. The total TTFT boundaries MUST remain unchanged. Admission MUST use order-balanced matched physical evidence and MUST reject missing, non-finite, inconsistent, or identity-mismatched observations.

#### Scenario: Phase evidence is complete
r[tenstorrent.qwen38_p150x2.traced_decode_ttft_evidence.valid]
- GIVEN matched physical responses with complete phase timings and table widths
- WHEN the deterministic receipt collector evaluates them
- THEN it MUST retain phase medians and total TTFT ratios
- AND admission MAY pass only when every existing correctness and performance gate passes.

#### Scenario: Phase evidence is incomplete
r[tenstorrent.qwen38_p150x2.traced_decode_ttft_evidence.invalid]
- GIVEN a missing phase, non-finite duration, negative duration, invalid width, or mismatched identity
- WHEN receipt validation runs
- THEN it MUST reject the candidate evidence
- AND production MUST remain eager.

#### Scenario: Physical candidate passes
r[tenstorrent.qwen38_p150x2.traced_decode_ttft_evidence.physical]
- GIVEN matched `eager, candidate, candidate, eager` physical roots on device IDs `0` and `1`
- WHEN exact parity, reset, decode, TTFT, tail, phase, and identity gates pass
- THEN traced decode MAY become physically admitted
- AND the admitted profile MUST bind the new receipt BLAKE3.
