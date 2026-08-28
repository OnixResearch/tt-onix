## ADDED Requirements

### Requirement: Persistent decode production telemetry
r[tenstorrent.rwkv7_p150x2.persistent_decode_telemetry] Every successful fast-worker receipt MUST contain one schema-versioned, aggregate-only persistent decode telemetry event. The event MUST record eligibility, selected dispatch mode, selected window, fallback reason, bounded timing values, terminal outcome counts, parity status, and cleanup status. The runtime MUST derive categorical values from a pure typed core. Evidence validation MUST reject telemetry that disagrees with the detailed worker receipt.

#### Scenario: Persistent execution emits telemetry
r[tenstorrent.rwkv7_p150x2.persistent_decode_telemetry.valid]
- GIVEN an eligible request that completes an admitted persistent window
- WHEN the fast worker writes its receipt
- THEN telemetry MUST report persistent mode, the selected window, no fallback, passed parity, and complete cleanup
- AND setup, replay, and cleanup durations MUST be finite and non-negative.

#### Scenario: Normal fallback emits telemetry
r[tenstorrent.rwkv7_p150x2.persistent_decode_telemetry.fallback]
- GIVEN a request that does not qualify for persistent dispatch
- WHEN the fast worker completes with normal dispatch
- THEN telemetry MUST report normal mode and the exact fallback reason
- AND trace timing, parity, cleanup, and terminal lifecycle fields MUST make no persistent-execution claim.

#### Scenario: Terminal event occurs before replay
r[tenstorrent.rwkv7_p150x2.persistent_decode_telemetry.terminal]
- GIVEN an admitted request with cancellation, timeout, or failure visible before replay
- WHEN the worker acknowledges the event and releases the slot
- THEN telemetry MUST report the exact terminal outcome count and complete cleanup
- AND telemetry MUST report zero trace replay time and no parity claim.

#### Scenario: Telemetry disagrees with evidence
r[tenstorrent.rwkv7_p150x2.persistent_decode_telemetry.invalid]
- GIVEN telemetry whose mode, window, fallback, parity, timing, terminal counts, or cleanup status disagrees with the detailed receipt
- WHEN the evidence guard validates the receipt
- THEN validation MUST fail with a typed telemetry error.

#### Scenario: Telemetry contains high-cardinality data
r[tenstorrent.rwkv7_p150x2.persistent_decode_telemetry.aggregate_only]
- GIVEN a telemetry event that contains prompt text, token IDs, request IDs, paths, binding identities, or unbounded labels
- WHEN the evidence guard validates the telemetry schema
- THEN validation MUST reject the event.
