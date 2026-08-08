# Tasks: Harness device utilization

## Phase 1: Utilization and memory

- [x] [serial] B1 Record baseline package tests and Cairn validation before edits. r[tenstorrent.harness.utilization]
- [x] [depends:tenstorrent.harness.repeat_stats] U1 Add the math or ALU busy fraction and the NoC idle fraction to the measurement run. r[tenstorrent.harness.utilization]
- [x] [depends:tenstorrent.harness.repeat_stats] U2 Add peak L1 or SRAM and DRAM or sysmem usage with total available capacity. r[tenstorrent.harness.memory_pressure]

  B1-U2 evidence: all three focused Nix checks and Cairn lifecycle gates passed before edits. The pure core now admits source-named math busy and NoC idle fractions plus per-device L1 and DRAM peaks and capacities.

## Phase 2: Op latency

- [x] [depends:tenstorrent.harness.utilization] O1 Add the per-operation latency table with mean and p99 latency for every operation in the step. r[tenstorrent.harness.op_latency]

  O1 evidence: the deterministic core requires exact executed-operation coverage and computes mean and p99 from positive profiler samples.

## Phase 3: Scaling efficiency

- [x] [depends:tenstorrent.harness.repeat_stats] E1 Add the one-device and two-device time-per-step measurement and the scaling ratio against the ideal data-parallel line. r[tenstorrent.harness.scaling_efficiency]

  E1 evidence: dual admission re-validates a compatible one-device measurement, binds both identities, and reports both step times, the ideal two-device line, and efficiency.

## Phase 4: Verification and documentation

- [x] [parallel] V1 Add positive fixtures for utilization, memory, op latency, and scaling ratio. r[tenstorrent.harness.utilization] r[tenstorrent.harness.op_latency]
- [x] [parallel] V2 Add negative fixtures for out-of-range utilization, invalid memory peak, a missing operation, and a scaling claim without a two-device run. r[tenstorrent.harness.memory_pressure] r[tenstorrent.harness.scaling_efficiency]
- [x] [serial] D1 Update each package README with a utilization and bottleneck section. r[tenstorrent.harness.utilization]

  V1-D1 evidence: 20 shared-core tests pass, both package checks admit their checked telemetry fixtures, and negative cases fail with field-specific diagnostics. Both READMEs document profiler conversion, required flags, receipt outputs, and claim limits.
- [x] [depends:tenstorrent.harness.op_latency] G1 Run formatting, package tests, Nix checks, Cairn validation, and all lifecycle gates. r[tenstorrent.harness.op_latency]

  G1 evidence: Rust formatting, Clippy with warnings denied, 20 shared-core tests, dual-core tests, shell syntax, all three focused Nix checks, and `nix run .#quality` pass. Final Cairn validation and all lifecycle gates pass.

## Verification Coverage

- `Scenario: Utilization fractions are recorded and bounded` -> U1, V1
- `Scenario: Out-of-range utilization is rejected` -> U1, V2
- `Scenario: Memory peaks are recorded against capacity` -> U2, V1
- `Scenario: Negative or non-finite memory peak is rejected` -> U2, V2
- `Scenario: Op latency table covers every operation` -> O1, V1
- `Scenario: A step operation missing from the table is rejected` -> O1, V2
- `Scenario: Scaling ratio is computed against the ideal line` -> E1, V1
- `Scenario: Scaling ratio without a two-device run is rejected` -> E1, V2
