# Tasks: Harness performance measurement

## Phase 1: Baseline and measurement profile

- [ ] [serial] B1 Record baseline package tests, Cairn validation, and the current parity receipts before edits. r[tenstorrent.harness.measurement_profile]
- [ ] [serial] M1 Add a named measurement profile that is distinct from the parity profile, with a repeat count and a metric set. r[tenstorrent.harness.measurement_profile]
- [ ] [depends:tenstorrent.harness.measurement_profile] M2 Add the stage time breakdown over launch, compile, probe or step, checkpoint, and teardown. r[tenstorrent.harness.stage_breakdown]

## Phase 2: Speedup and machine facts

- [ ] [depends:tenstorrent.harness.measurement_profile] S1 Add the measured CPU-reference stage time under the same harness and compute the speedup ratio into the receipt. r[tenstorrent.harness.speedup_ratio]
- [ ] [depends:tenstorrent.harness.measurement_profile] S2 Bind board, firmware, PCIe, power state, and mesh machine facts to each receipt. r[tenstorrent.harness.machine_facts]

## Phase 3: Repeat statistics

- [ ] [depends:tenstorrent.harness.measurement_profile] R1 Implement the M-repeat run mode and report mean, p50, p99, and standard deviation for time and loss. r[tenstorrent.harness.repeat_stats]

## Phase 4: Verification and documentation

- [ ] [parallel] V1 Add positive fixtures for measurement profile export, speedup ratio, machine facts, and repeat statistics. r[tenstorrent.harness.measurement_profile] r[tenstorrent.harness.repeat_stats]
- [ ] [parallel] V2 Add negative fixtures for a parity-relaxing profile, a non-finite stage time, a missing CPU baseline, missing machine facts, and a low repeat count. r[tenstorrent.harness.stage_breakdown] r[tenstorrent.harness.machine_facts]
- [ ] [serial] D1 Update each package README with the measurement section and revise the non-claims section. r[tenstorrent.harness.measurement_profile]
- [ ] [depends:tenstorrent.harness.repeat_stats] G1 Run formatting, package tests, Nix checks, Cairn validation, and all lifecycle gates with no parity regression. r[tenstorrent.harness.measurement_profile]

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
