# RWKV-7 P150x2 runtime delta

## ADDED Requirements

### Requirement: Fused-only window-16 diagnostic

r[tenstorrent.rwkv7_p150x2.window16_fused_only_diagnostic]

The diagnostic runtime MUST compare the asymmetric fused WKV result against a pure host reference without running generic TT-NN recurrence or correction operations before fused execution. It MUST bind the receipt to the compiled runtime identity and readout precision setting.

#### Scenario: Diagnostic reaches exact physical parity

- **GIVEN** the diagnostic-only BF16 readout runtime and asymmetric fixture
- **WHEN** the bounded physical probe executes on device `0`
- **THEN** the receipt proves exact FP32 recurrent-state and BF16 corrected-row parity against the pure host reference

#### Scenario: Wrong runtime or pre-work is present

- **GIVEN** a launcher that resolves to another runtime or a probe that runs generic device operations before fused execution
- **WHEN** diagnostic admission validates the execution plan
- **THEN** it rejects the run before making a window-16 correction claim

### Requirement: Window-16 reevaluation remains separate

r[tenstorrent.rwkv7_p150x2.window16_reevaluation_after_diagnostic]

A passing fused-only diagnostic MUST NOT modify production admission. A later window-16 evaluation MUST create a new immutable candidate and repeat every required simulator and physical gate.

#### Scenario: Diagnostic parity passes

- **GIVEN** an exact physical diagnostic receipt
- **WHEN** operators prepare window-16 reevaluation
- **THEN** they create a new immutable candidate and start all gates from the beginning

#### Scenario: Diagnostic fails or times out

- **GIVEN** a mismatch, missing receipt, wrong package binding, or bounded timeout
- **WHEN** the diagnostic ends
- **THEN** later admission gates do not start and production remains `[2, 4, 8]`
