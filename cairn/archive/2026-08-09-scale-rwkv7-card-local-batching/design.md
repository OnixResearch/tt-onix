## Context

Each P150 card runs one complete RWKV-7 replica. Each live request owns FP32 recurrent state and one deterministic scheduler slot. The current constant permits four slots per card and at most two bounded waves.

The model checkpoint is about 382 MB. Larger card-local batches can reuse loaded weight tiles across more requests. They also increase recurrent-state storage, temporary tensor storage, queue depth, and tail-latency risk.

The goal is not to promise a rate. The goal is to find the largest useful capacity through a bounded physical sweep.

## Goal and Success Contract

The implementation is complete when:

- B4, B16, B32, and B64 plans pass deterministic device-free validation.
- Unsafe memory plans fail before a device opens.
- Every live request has one unique card-local state slot.
- Simulator smoke passes for the production binary at the selected candidate capacity.
- Every physical candidate preserves exact FP32-reference generated tokens.
- A candidate is promoted only after stable alternating A/B evidence and strict two-card admission pass.
- A miss records the measured blocker and leaves the last admitted capacity active.

A passing capacity receipt does not prove long-duration stability or `1,000` tokens per second.

## Decisions

### Decision: Put capacity in the typed profile

**Choice:** Nickel defines candidate capacities, memory bounds, repeat counts, workload shape, and admission thresholds. Runtime code consumes exported values.

**Rationale:** Capacity depends on workload and evidence policy. It is not a universal runtime constant.

### Decision: Keep scheduling logic pure

**Choice:** A pure planner maps ready requests to cards, slots, and bounded waves. The shell allocates tensors, opens devices, runs workers, and writes receipts.

**Rationale:** Slot uniqueness, ordering, overflow, and memory arithmetic can be tested without hardware.

### Decision: Derive memory before allocation

**Choice:** The planner computes FP32 recurrent-state bytes, token-shift bytes, temporary tensor estimates, and a configured safety reserve. It rejects overflow and insufficient capacity before device access.

**Rationale:** Large batches must fail clearly. Device allocation failure is not an admission strategy.

### Decision: Sweep powers of four without assuming linear scaling

**Choice:** Measure B4, B16, B32, and B64 in that order. Stop increasing capacity after an out-of-memory result, parity failure, repeated latency-bound failure, or throughput regression.

**Rationale:** Larger batches can improve weight reuse, but device step time and tail latency can grow.

### Decision: Preserve deterministic request ownership

**Choice:** Slot IDs remain stable for each live request. Completed slots return to a sorted free list. A new request cannot observe prior state.

**Rationale:** Capacity growth must not introduce cross-request state leakage or order-dependent output.

### Decision: Use physical evidence for promotion

**Choice:** Compare each candidate with the current admitted runtime through order-balanced A/B runs. Then run strict one-card and two-card admission.

**Rationale:** Launch reduction, occupancy, and simulator success are not throughput proof.

## Validation Strategy

Positive tests cover all four candidate plans, deterministic assignments, slot recycling after zeroing, exact memory arithmetic, and admitted receipt calculation.

Negative tests cover duplicate ownership, stale state reuse, unsupported capacity, arithmetic overflow, insufficient memory reserve, malformed metrics, non-finite metrics, parity mismatch, and a simulator speed claim.

The simulator smoke uses the production binary and package-matched Blackhole simulator. Physical tests use isolated single-card workers and exact card attribution.

## Risks and Trade-offs

- **Memory pressure:** derive and check all bounded allocations before device open.
- **Tail latency:** retain profile-owned p99 limits and reject regressions.
- **Longer compile time:** exclude compile and warm-up from decode timing.
- **State leakage:** zero recycled slots and test negative reuse cases.
- **Scheduler overhead:** measure host gaps and queue occupancy.
- **Nonstationary hardware:** use alternating A/B runs before strict admission.

## Non-Goals

- A single-request latency improvement.
- Tensor parallel execution.
- A lower-precision model.
- A throughput claim from ideal batch multiplication.
