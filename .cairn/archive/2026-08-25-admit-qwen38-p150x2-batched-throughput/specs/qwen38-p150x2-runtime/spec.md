## ADDED Requirements

### Requirement: Batched throughput admission uses matched physical roots

r[tenstorrent.qwen38_p150x2.batch_performance_admission]

The repository MUST define batch admission in a closed typed profile. The profile MUST compare the same traced production path at widths B=1 and B=4. Physical execution MUST use devices 0 and 1 in exact `single,batch,batch,single` root order. Each root MUST use the same prompt corpus, generation count, package, runtime, checkpoint, tokenizer, and mesh. Admission MUST require exact per-request token parity, a complete active state prefix, distinct state slices, clean reset, aggregate decode speedup at least `3.0`, per-user decode-rate retention at least `0.75`, TTFT ratio at most `4.5`, and tail-latency ratio at most `1.5`. The existing single-stream admission profile and its `1.05` TTFT gate MUST remain unchanged.

#### Scenario: B=4 passes every batch gate

- GIVEN complete current-identity B=1 and B=4 physical roots with matching requests
- WHEN the deterministic batch receipt evaluator checks correctness, isolation, timing, throughput, and cleanup
- THEN it MUST admit only when every profile-owned bound passes
- AND the admitted profile MUST bind the receipt BLAKE3 identity.

#### Scenario: Batch evidence is missing or invalid

- GIVEN a missing root, wrong order, identity drift, token mismatch, state collision, incomplete active prefix, invalid timing, cleanup failure, or failed performance bound
- WHEN batch admission evaluates the evidence
- THEN it MUST reject the complete candidate with bounded reasons
- AND it MUST preserve `physically_admitted = false`.

### Requirement: Batch TTFT excludes trace preparation

r[tenstorrent.qwen38_p150x2.batch_ttft_evidence]

Batch TTFT MUST start immediately before production prefill dispatch and stop after every active user's first-token winner is available. Exact-width trace preparation MUST occur after that boundary and MUST remain in a separate evidence field. Queue collection, later decode steps, and cleanup MUST NOT enter the internal TTFT phase. Normal production and soak responses MUST remain admission-ineligible. Only an explicit operator batch-admission mode MAY emit admission-eligible evidence.

#### Scenario: Batch phase evidence is complete

- GIVEN an explicit batch-admission run with valid B=1 or B=4 requests
- WHEN prefill, first-token selection, trace preparation, decode, and cleanup complete
- THEN each response MUST report finite non-negative separated phase durations
- AND its TTFT MUST exclude trace preparation while retaining first-token work.

#### Scenario: Ordinary traffic reaches the receipt evaluator

- GIVEN a normal production, readiness, soak, profiled, or simulated response
- WHEN the batch receipt evaluator reads it
- THEN it MUST reject the response as admission-ineligible
- AND it MUST not infer admission from a throughput value.

## MODIFIED Requirements

### Requirement: Legacy continuous batching remains unsupported without separate admission

r[tenstorrent.qwen38_p150x2.continuous_batching_non_support]

The repository MUST NOT treat the bounded active-prefix batch path as proof of a general continuous scheduler. Arbitrary slot recycling, mixed generation bounds, non-prefix occupancy, live prefill during decode, and dynamic batch compaction remain unsupported until separate typed changes admit them. The older unadmitted single-stream trace receipt MUST NOT override the later bounded batching requirements. A batch-specific receipt MAY admit only the exact width, workload, state layout, and identity that it evaluates.

#### Scenario: Bounded active-prefix batching is evaluated

- GIVEN the accepted isolated batch path and its batch-specific typed profile
- WHEN exact B=1 versus B=4 physical roots are evaluated
- THEN the receipt MAY decide only bounded B=4 throughput admission
- AND it MUST NOT claim general continuous batching.

#### Scenario: An operator requests unsupported scheduling

- GIVEN mixed generation bounds, non-prefix occupancy, slot recycling, live prefill during decode, or dynamic compaction
- WHEN request execution evaluates the unsupported schedule
- THEN the service MUST reject or defer that schedule before state mutation
- AND the bounded B=4 receipt MUST NOT authorize it.
