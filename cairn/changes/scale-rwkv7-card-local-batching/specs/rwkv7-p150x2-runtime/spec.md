# RWKV-7 P150x2 Runtime Specification Delta

## ADDED Requirements

### Requirement: Scalable card-local batching
r[tenstorrent.rwkv7_p150x2.scalable_card_local_batching] The fast runtime MUST derive card-local request capacity from a typed profile and a checked memory plan. It MUST preserve unique request state, FP32 recurrence, deterministic scheduling, and exact generated-token parity at every promoted capacity. A capacity above four MUST NOT become the default without stable physical throughput evidence and strict two-card admission.

#### Scenario: Larger capacity passes
r[tenstorrent.rwkv7_p150x2.scalable_card_local_batching.valid]
- GIVEN a supported candidate capacity with sufficient checked device memory
- AND every live request owns one unique zero-initialized card-local state slot
- WHEN simulator, exact-parity, alternating A/B, and strict physical admission complete
- THEN the runtime MAY promote that capacity
- AND the receipt MUST report capacity, effective batch size, memory use, throughput, latency, card attribution, and evidence identities.

#### Scenario: Unsafe capacity fails before device access
r[tenstorrent.rwkv7_p150x2.scalable_card_local_batching.invalid_memory]
- GIVEN a capacity plan with arithmetic overflow, insufficient memory reserve, or an unsupported capacity
- WHEN capacity admission runs
- THEN admission MUST fail before a device opens
- AND it MUST identify the rejected resource bound.

#### Scenario: State ownership is invalid
r[tenstorrent.rwkv7_p150x2.scalable_card_local_batching.invalid_state]
- GIVEN a schedule with duplicate live-slot ownership, stale recycled state, or cross-card request ownership
- WHEN the pure scheduler validates the plan
- THEN validation MUST fail before decode dispatch.

#### Scenario: Throughput target is missed
r[tenstorrent.rwkv7_p150x2.scalable_card_local_batching.target_miss]
- GIVEN a larger capacity that fails exact parity, stability, throughput, or p99 latency admission
- WHEN the bounded sweep evaluates the candidate
- THEN the runtime MUST keep the last admitted capacity
- AND the evidence package MUST retain the measured blocker without claiming the target rate.
