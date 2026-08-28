# RWKV-7 P150x2 Runtime Specification Delta

## ADDED Requirements

### Requirement: Per-token capacity admission
r[tenstorrent.rwkv7_p150x2.scalable_card_local_batching.per_token_admission] The capacity admission gate MUST compare each candidate against the current admitted capacity on p99 time per generated token, computed as the p99 of (sample elapsed divided by tokens served in that sample). Per-request wall latency MUST NOT be the admission basis across different capacity workloads, because it grows with batch concurrency and does not measure per-token device cost. A promoted candidate MUST keep the strict physical, profiler, parity, and occupancy rails from the parent requirement.

#### Scenario: Candidate faster per token is promoted
r[tenstorrent.rwkv7_p150x2.scalable_card_local_batching.per_token_admission.valid]
- GIVEN a candidate measured on physical cards with exact FP32-reference parity and profiler attribution
- AND the candidate's aggregate tokens per second exceeds the current admitted capacity
- AND the candidate's p99 time per generated token does not regress beyond the profile bound
- WHEN the pure gate evaluates the order-balanced evidence
- THEN the capacity MAY be promoted
- AND the evidence package MUST retain the order-balanced runs and the admission decision.

#### Scenario: Metric basis is rejected
r[tenstorrent.rwkv7_p150x2.scalable_card_local_batching.per_token_admission.invalid]
- GIVEN evidence whose latency admission is reported as per-request wall latency across different capacity workloads
- WHEN the gate validates the admission basis
- THEN the gate MUST reject that basis
- AND the change MUST record the corrected per-token comparison without claiming the wall-latency basis was meaningful.
