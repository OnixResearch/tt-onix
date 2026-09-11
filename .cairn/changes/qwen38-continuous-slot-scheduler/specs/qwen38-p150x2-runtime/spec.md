## ADDED Requirements

### Requirement: Deterministic bounded slot lifecycle

r[tenstorrent.qwen38_p150x2.continuous_slot_policy]

The pure scheduler MUST own slot generation, request identity, position, deadline, completion, and refill decisions. It MUST support unequal generation bounds without reassigning another user's state. Initial admission MUST remain within B4 capacity. Queue and memory limits MUST come from typed policy.

#### Scenario: Staggered requests
- GIVEN one completed slot and another active request
- WHEN a compatible queued request is eligible
- THEN the scheduler plans refill without changing the active request's identity or position.

#### Scenario: Stale event or full queue
- GIVEN an old slot-generation event or exhausted queue capacity
- WHEN the scheduler evaluates the event
- THEN it rejects the event or request without corrupting active state.

### Requirement: Single device owner and safe refill

r[tenstorrent.qwen38_p150x2.continuous_slot_effects]

One shell worker MUST serialize all device effects across serving routes. Refill MUST preserve active KV, recurrent, and convolution state. Trace addresses and inactive-row behavior MUST satisfy an explicit allocation contract. An unsafe prefill transition MUST use a drained fallback rather than replay a suspect trace.

#### Scenario: Safe slot refill
- GIVEN a tested refill plan and stable state buffers
- WHEN the adapter installs a new prefix
- THEN existing users continue with the same reference state and no address drift.

#### Scenario: Timeout or allocation drift
- GIVEN a timed-out worker or invalid trace binding
- WHEN the shell observes the error
- THEN it quarantines the engine and prevents device reuse.

### Requirement: Queue-inclusive scheduling admission

r[tenstorrent.qwen38_p150x2.continuous_slot_admission]

The scheduling evaluator MUST compare staggered mixed-length workloads against the current group coordinator. It MUST include queue time, prefill interference, capture, tail latency, fairness, and delivered-token throughput. Reduced capture count alone MUST NOT satisfy admission.

#### Scenario: Mixed-length benefit
- GIVEN matched workloads with short and long requests
- WHEN every latency, fairness, correctness, and throughput gate passes
- THEN the receipt admits the measured scheduler profile only.

#### Scenario: Starvation or hidden queue delay
- GIVEN higher aggregate decode but starvation or worse declared tail latency
- WHEN the evaluator applies the policy
- THEN it rejects the candidate.
