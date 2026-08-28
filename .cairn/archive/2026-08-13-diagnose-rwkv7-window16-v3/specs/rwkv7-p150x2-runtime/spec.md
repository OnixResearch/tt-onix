## ADDED Requirements

### Requirement: Window-16 full-model diagnosis

r[tenstorrent.rwkv7_p150x2.window16_full_model_diagnosis]

The runtime MUST provide a bounded normal-dispatch diagnosis for the known window-16 CPU parity mismatch. It MUST compare package-matched generic TT-NN and fused execution at deterministic layer and operation boundaries. It MUST retain aggregate hashes, maximum errors, and the first divergent boundary only. It MUST NOT create or admit candidate v3 before exact pinned CPU generated-token and FP32 recurrent-state parity pass.

#### Scenario: The diagnosis finds a first divergent boundary

- **GIVEN** the immutable rejected v2 identity and the known failing prefix
- **WHEN** one bounded normal-dispatch diagnosis completes
- **THEN** the receipt identifies the first divergent token, layer, operation, and compared execution paths
- **AND** the receipt includes deterministic hashes and finite aggregate errors
- **AND** no raw tensor, model cache, compiler output, or profiler output is retained

#### Scenario: The diagnosis does not establish a root cause

- **GIVEN** incomplete evidence, a busy device, a timeout, or multiple indistinguishable mechanisms
- **WHEN** the diagnosis classifies its result
- **THEN** it stops without creating candidate v3
- **AND** production admission remains `[2, 4, 8]`

#### Scenario: A correction passes exact parity

- **GIVEN** one proven root cause and its bounded correction
- **WHEN** the sixteen-token physical correction probe completes
- **THEN** every generated token matches the pinned CPU continuation exactly
- **AND** the final recurrent state matches the pinned CPU FP32 authority exactly
- **AND** a new immutable candidate can bind the diagnostic receipt for later admission gates
