## MODIFIED Requirements

### Requirement: Persistent decode window 16

r[tenstorrent.rwkv7_p150x2.persistent_decode_window_16]

Window `16` MUST use a new immutable candidate after the fused WKV correction. The candidate MUST preserve exact CPU generated-token and FP32 recurrent-state parity. It MUST pass simulator, physical lifecycle, profiler, A/B, canary, monitoring, deployment, and observation gates before production admission.

#### Scenario: Corrected v2 candidate passes complete evidence

- **GIVEN** the immutable v2 candidate and unowned physical devices `0` and `1`
- **WHEN** every bounded gate completes
- **THEN** every accepted result has exact CPU generated-token and FP32 recurrent-state parity
- **AND** every request is acknowledged, zeroed, and released
- **AND** each one-card and two-card paired speedup is at least `1.0`
- **AND** canary, monitoring, deployment, and observation evidence is clean

#### Scenario: Rejected v1 identity is reused

- **GIVEN** the rejected v1 profile, runtime, evidence package, hash, or receipt
- **WHEN** v2 admission validates its identity chain
- **THEN** admission fails before simulator or physical execution

#### Scenario: An evidence gate fails

- **GIVEN** a profile error, parity mismatch, cleanup error, profiler drop, speed loss, timeout, canary error, deployment error, or monitoring alert
- **WHEN** the evidence is classified
- **THEN** all later gates stop
- **AND** production admission remains `[2, 4, 8]`
