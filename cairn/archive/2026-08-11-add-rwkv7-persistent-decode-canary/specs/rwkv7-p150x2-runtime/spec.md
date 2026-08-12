# RWKV-7 P150x2 Runtime Delta

## ADDED Requirements

### Requirement: Bounded persistent decode canary
r[tenstorrent.rwkv7_p150x2.persistent_decode_canary] The system MUST provide a bounded physical canary controller whose BLAKE3-bound runtime interlock forces normal dispatch after every held, disabled, malformed, mismatched, or critical canary state.

#### Scenario: Canary policy is valid
r[tenstorrent.rwkv7_p150x2.persistent_decode_canary.policy]
- GIVEN the strict production canary policy and matched throughput profile
- WHEN Nickel exports them and Rust validates their cross-field budgets
- THEN machine, device, request, duration, traffic, scheduling, profile, window, and gain limits agree

#### Scenario: Armed interlock permits candidate execution
r[tenstorrent.rwkv7_p150x2.persistent_decode_canary.interlock]
- GIVEN an armed interlock with matching file BLAKE3, canary binding, profile identity, and bounded limits
- WHEN an eligible canary worker selects dispatch
- THEN persistent trace remains available for the candidate worker

#### Scenario: Runtime receives a blocked interlock
r[tenstorrent.rwkv7_p150x2.persistent_decode_canary.runtime]
- GIVEN a held, disabled, malformed, hash-mismatched, binding-mismatched, or profile-mismatched interlock
- WHEN any worker requests persistent dispatch, including a candidate override
- THEN the runtime uses normal dispatch with `canary_interlock_blocked`

#### Scenario: Canary evidence is evaluated
r[tenstorrent.rwkv7_p150x2.persistent_decode_canary.evaluate]
- GIVEN a bounded completed benchmark, strict candidate telemetry, and matched physical A/B evidence
- WHEN the pure canary core evaluates parity, cleanup, terminal outcomes, gain, duration, and request budgets
- THEN it emits a deterministic final interlock and canary receipt

#### Scenario: Operator runs the canary command
r[tenstorrent.rwkv7_p150x2.persistent_decode_canary.command]
- GIVEN explicit policy, throughput, workload, monitoring, and new output-root paths
- WHEN the operator runs the packaged canary command
- THEN it arms the interlock, runs one matched benchmark under the policy timeout, and atomically writes the final state
- AND one-card samples balance concurrent workers across both cards while dual-card samples alternate order

#### Scenario: Canary evidence is invalid or unsafe
r[tenstorrent.rwkv7_p150x2.persistent_decode_canary.invalid]
- GIVEN a policy mismatch, request overflow, timeout, benchmark failure, invalid A/B receipt, warning, critical alert, parity failure, cleanup failure, or terminal failure
- WHEN the controller evaluates the run
- THEN it does not emit `retain`
- AND every critical or invalid result emits `disable`

#### Scenario: Physical canary passes
r[tenstorrent.rwkv7_p150x2.persistent_decode_canary.physical]
- GIVEN one bounded run on physical devices `0` and `1`
- WHEN matched normal and persistent samples complete with clean candidate monitoring
- THEN the retained canary receipt proves exact parity, complete cleanup, stable positive gain, bounded duration, and final `retain`
