## MODIFIED Requirements

### Requirement: Fused-only window-16 diagnostic

r[tenstorrent.rwkv7_p150x2.window16_fused_only_diagnostic]

The diagnostic runtime MUST run its finite fused programs before package-matched generic TT-NN recurrence or correction. It MUST use generic TT-NN as the exact device authority. A scalar host calculation MAY provide a value summary, but it MUST NOT be treated as package-bit-exact. The receipt MUST bind the compiled runtime identity, recurrence precision, readout precision, and correction owner.

#### Scenario: Diagnostic reaches exact physical parity

- **GIVEN** the diagnostic-only split runtime and asymmetric fixture
- **WHEN** the bounded physical probe executes on device `0`
- **THEN** every FP32 state value and every raw and corrected BF16 row bit matches package-matched generic TT-NN exactly
- **AND** the receipt makes no scalar-host, performance, or production admission claim

#### Scenario: Wrong runtime or pre-work is present

- **GIVEN** a launcher that resolves to another runtime or runs generic device operations before fused execution
- **WHEN** diagnostic validation reads the execution plan or receipt
- **THEN** it rejects the run before making a window-16 correction claim

#### Scenario: Scalar host values differ from generic TT-NN

- **GIVEN** scalar host values that are not package-bit-exact
- **WHEN** the diagnostic receipt is evaluated
- **THEN** the values remain a non-authoritative summary
- **AND** exact device parity is decided only against package-matched generic TT-NN

## ADDED Requirements

### Requirement: Bounded fused WKV completion diagnosis

r[tenstorrent.rwkv7_p150x2.window16_fused_completion]

The diagnostic runtime MUST execute a finite fused WKV path. It MUST compare the complete recurrent state, raw readout, and corrected rows with package-matched generic TT-NN. Diagnostic precision, correction ownership, source identity, and writer completion MUST fail closed. The diagnostic path MUST NOT change production admission.

#### Scenario: Diagnostic execution does not complete

- **GIVEN** the immutable diagnostic runtime and asymmetric fixture
- **WHEN** a launch, synchronized readback, correction, or receipt stage does not finish within its named bound
- **THEN** no exact completion claim is accepted
- **AND** no later window-16 admission gate starts

#### Scenario: Full split execution completes

- **GIVEN** HiFi4 FP32 recurrence and a separate HiFi2 BF16 readout program
- **WHEN** both programs finish and synchronized readback completes
- **THEN** every FP32 recurrent-state value matches package-matched generic TT-NN exactly
- **AND** every raw and corrected BF16 row bit matches package-matched generic TT-NN exactly
- **AND** the caller applies correction exactly once after canonical row reshape

#### Scenario: Diagnostic contract drifts

- **GIVEN** a wrong source order, fidelity, accumulator width, CB format, correction owner, row geometry, runtime identity, or writer signal
- **WHEN** the device-free source contracts run
- **THEN** they reject the diagnostic path before a physical completion claim

#### Scenario: Production runtime is built

- **GIVEN** the accepted production runtime
- **WHEN** its fused kernels compile
- **THEN** split FP32 recurrence and BF16 readout mode is disabled
- **AND** production windows remain `[2, 4, 8]`
