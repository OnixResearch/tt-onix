## ADDED Requirements

### Requirement: Width sixteen executes and is measured, or its blocker is exact

r[tenstorrent.qwen38_p150x2.b16_execution]

The service's declared maximum batch width MUST be reachable through the same
closed policy the other widths use. A B=16 run MUST carry its own profile with
the shared bounds, a sixteen-prompt corpus, and a physical-evidence binary. A
B=16 measurement MUST record either a receipt covering every gate or the exact
failing gate and the exact program defect that prevents one. The change MUST NOT
enable batched serving in production and MUST NOT move the B=4 or B=8 bindings.

#### Scenario: Sixteen concurrent users execute

r[tenstorrent.qwen38_p150x2.b16_execution.valid]

- **GIVEN** the widened policy, the B=16 profile, and the sixteen-prompt corpus
- **WHEN** sixteen concurrent requests run on physical devices 0 and 1
- **THEN** every request MUST return at `batch_width` 16
- **AND** the receipt MUST record the instance count, the core count, and the
  chunk length the chunked scan used
- **AND** exact token and text parity MUST hold against the single-stream roots.

#### Scenario: The width is refused when it cannot be trusted

r[tenstorrent.qwen38_p150x2.b16_execution.invalid]

- **GIVEN** a width outside the widened closed set, or a profile whose bounds,
  corpus, or order differ from the shared policy
- **WHEN** the profile or the receipt contracts are validated
- **THEN** validation MUST fail before any device work
- **AND** no partially valid B=16 profile MAY load.

#### Scenario: A program defect prevents a receipt

r[tenstorrent.qwen38_p150x2.b16_execution.blocked]

- **GIVEN** a physical B=16 run that fails with a program error
- **WHEN** the failure repeats or is recovered only by a whole-group retry
- **THEN** the receipt MUST NOT be admitted
- **AND** the exact diagnostic, the affected program, and the observed rate MUST
  be recorded as the blocker.
