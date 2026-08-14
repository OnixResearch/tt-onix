## ADDED Requirements

### Requirement: FP32 recurrent-matrix diagnosis

r[tenstorrent.rwkv7_p150x2.fp32_recurrent_matrix_diagnosis]

The runtime MUST provide a bounded diagnosis for the window-16 FP32 recurrent-matrix mismatch. It MUST compare the pinned CPU authority with the high-precision BF16 model-matmul, generic-WKV, normal-dispatch device path. It MUST identify the first differing model step, layer, and recurrence operation with deterministic identities and finite aggregate errors. It MUST retain aggregate evidence only. It MUST NOT create or admit candidate v3 before exact generated-token and FP32 recurrent-matrix parity pass.

#### Scenario: The diagnosis finds the first recurrent-matrix divergence

- **GIVEN** the pinned CPU authority and the high-precision device control
- **WHEN** one bounded diagnosis compares recurrent-matrix boundaries
- **THEN** the receipt identifies the first differing model step, layer, and recurrence operation
- **AND** the receipt includes the data type, element count, first differing element, finite maximum error, and deterministic CPU and device identities
- **AND** no raw tensor, cache, compiler output, profiler output, or operational log is retained

#### Scenario: A mechanism remains unresolved

- **GIVEN** a busy device, a timeout, unmatched harnesses, or indistinguishable numerical mechanisms
- **WHEN** the bounded diagnosis cannot prove one cause
- **THEN** it stops with the exact blocker or exhausted registry
- **AND** candidate v3 remains absent
- **AND** production admission remains `[2, 4, 8]`

#### Scenario: A bounded correction passes exact parity

- **GIVEN** one proven recurrent-matrix divergence mechanism and its bounded correction
- **WHEN** the sixteen-token physical correction probe completes
- **THEN** every generated token matches the pinned CPU continuation exactly
- **AND** the final recurrent matrix matches the pinned CPU FP32 authority exactly
- **AND** a new immutable candidate can bind the diagnostic receipt for later admission gates

#### Scenario: A bounded correction fails the initial boundary

- **GIVEN** one proven recurrent-matrix divergence mechanism and its bounded correction
- **WHEN** the correction differs at the initial recurrent boundary
- **THEN** the sixteen-token correction probe does not start
- **AND** candidate v3 remains absent
- **AND** production admission remains `[2, 4, 8]`
- **AND** later performance, canary, deployment, observation, and admission gates do not start
