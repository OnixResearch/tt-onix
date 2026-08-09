# RWKV-7 P150x2 Runtime Specification Delta

## ADDED Requirements

### Requirement: Bounded persistent device decode
r[tenstorrent.rwkv7_p150x2.persistent_device_decode] The fast runtime MAY provide a persistent device decode mode that carries selected token IDs and FP32 recurrent state across bounded decode steps. The mode MUST validate all binding identities before start, MUST provide bounded completion and cancellation, and MUST perform no per-token host submission inside its admitted persistent region. The runtime MUST retain normal dispatch as a fail-closed fallback.

#### Scenario: Persistent decode passes
r[tenstorrent.rwkv7_p150x2.persistent_device_decode.valid]
- GIVEN valid runtime, checkpoint, program, weight, token, state, slot, output, and profile identities
- AND a request within the profile-owned step, time, and output bounds
- WHEN persistent decode executes
- THEN the device MUST carry the selected token and FP32 recurrent state between steps
- AND the host MUST perform no per-token submission inside the persistent region
- AND generated tokens MUST exactly match normal dispatch and the pinned FP32 CPU reference.

#### Scenario: Binding identity is stale
r[tenstorrent.rwkv7_p150x2.persistent_device_decode.stale_binding]
- GIVEN a stale or mismatched token, state, slot, output, weight, program, runtime, checkpoint, or profile identity
- WHEN persistent capability validation runs
- THEN it MUST reject persistent execution before the device loop starts
- AND the runtime MUST retain normal dispatch.

#### Scenario: Cancellation or timeout occurs
r[tenstorrent.rwkv7_p150x2.persistent_device_decode.bounded_exit]
- GIVEN an active persistent request that receives cancellation or reaches a profile-owned bound
- WHEN the device and host process the bounded exit
- THEN they MUST reach one deterministic terminal state
- AND the slot MUST NOT return to the free list until completion acknowledgement and state zeroing finish.

#### Scenario: Host submissions remain in the token loop
r[tenstorrent.rwkv7_p150x2.persistent_device_decode.not_persistent]
- GIVEN a candidate that still performs one or more host submissions for each token inside the measured region
- WHEN persistent-mode admission evaluates the profiler evidence
- THEN admission MUST fail
- AND the runtime MUST NOT describe the candidate as persistent decode.

#### Scenario: Persistent path has no stable gain
r[tenstorrent.rwkv7_p150x2.persistent_device_decode.no_gain]
- GIVEN a parity-correct persistent candidate without stable physical throughput improvement
- WHEN runtime selection evaluates the candidate
- THEN the runtime MUST keep normal dispatch as the default
- AND the evidence package MUST retain the measured blocker without claiming improvement.
