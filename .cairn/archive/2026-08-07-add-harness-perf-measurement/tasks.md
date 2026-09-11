# Tasks: Harness performance measurement

## Phase 1: Baseline and measurement profile

- [x] [serial] B1 Record baseline package tests, Cairn validation, and the current parity receipts before edits. r[tenstorrent.harness.measurement_profile]

  B1 evidence: the single-card, dual-card, and CPU-reference Nix checks passed before edits. Cairn validation and all three lifecycle gates passed. Existing parity profile identities remained `d6e9ec5e...` and `34d5ee998...` after implementation.
- [x] [serial] M1 Add a named measurement profile that is distinct from the parity profile, with a repeat count and a metric set. r[tenstorrent.harness.measurement_profile]
- [x] [depends:tenstorrent.harness.measurement_profile] M2 Add the stage time breakdown over launch, compile, probe or step, checkpoint, and teardown. r[tenstorrent.harness.stage_breakdown]

  M1-M2 evidence: both packages export deterministic measurement profiles without changing parity profiles. The runners partition host elapsed time across launch, compile, step, checkpoint, and teardown.

## Phase 2: Speedup and machine facts

- [x] [depends:tenstorrent.harness.measurement_profile] S1 Add the measured CPU-reference stage time under the same harness and compute the speedup ratio into the receipt. r[tenstorrent.harness.speedup_ratio]
- [x] [depends:tenstorrent.harness.measurement_profile] S2 Bind board, firmware, PCIe, power state, and mesh machine facts to each receipt. r[tenstorrent.harness.machine_facts]

  S1-S2 evidence: the pure core binds the CPU baseline and machine facts with BLAKE3. The runner rejects missing fact values before device access.

## Phase 3: Repeat statistics

- [x] [depends:tenstorrent.harness.measurement_profile] R1 Implement the M-repeat run mode and report mean, p50, p99, and standard deviation for time and loss. r[tenstorrent.harness.repeat_stats]

  R1 evidence: measurement mode runs one warm-up compile step and five measured steps. The deterministic core computes mean, p50, p99, and population standard deviation.

## Phase 4: Verification and documentation

- [x] [parallel] V1 Add positive fixtures for measurement profile export, speedup ratio, machine facts, and repeat statistics. r[tenstorrent.harness.measurement_profile] r[tenstorrent.harness.repeat_stats]
- [x] [parallel] V2 Add negative fixtures for a parity-relaxing profile, a non-finite stage time, a missing CPU baseline, missing machine facts, and a low repeat count. r[tenstorrent.harness.stage_breakdown] r[tenstorrent.harness.machine_facts]

  V1-V2 evidence: 12 single-core unit tests pass. Both package checks pass positive CLI fixtures and reject parity drift, invalid metrics, missing CPU identity, missing facts, and low repeat counts.
- [x] [serial] D1 Update each package README with the measurement section and revise the non-claims section. r[tenstorrent.harness.measurement_profile]

  D1 evidence: both package READMEs document measurement profiles, machine-fact input, commands, receipts, and claim boundaries.
- [x] [depends:tenstorrent.harness.repeat_stats] G1 Run formatting, package tests, Nix checks, Cairn validation, and all lifecycle gates with no parity regression. r[tenstorrent.harness.measurement_profile]

  G1 evidence: `nix run .#quality` passed after Nixfmt and Ruff formatting. Single-card, dual-card, and CPU-reference checks passed. Cairn validation and proposal, design, and tasks gates passed. Existing parity profile identities stayed unchanged.

## Verification Coverage

- `Scenario: Measurement profile exports deterministic metrics` -> M1, M2, V1
- `Scenario: Measurement profile cannot gate parity` -> M1, V2
- `Scenario: Stage times are recorded and bounded` -> M2, V1
- `Scenario: Negative or non-finite stage time is rejected` -> M2, V2
- `Scenario: Speedup ratio is computed against a bound baseline` -> S1, V1
- `Scenario: Speedup ratio without a CPU baseline is rejected` -> S1, V2
- `Scenario: Machine facts are bound to the receipt` -> S2, V1
- `Scenario: Missing machine facts fail admission` -> S2, V2
- `Scenario: Repeated run reports a distribution` -> R1, V1
- `Scenario: Repeats below the bound are rejected` -> R1, V2
