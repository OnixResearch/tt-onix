# Tasks: Harness stability and training value

## Phase 1: Long-run mode

- [ ] [serial] B1 Record baseline package tests and Cairn validation before edits. r[tenstorrent.harness.long_run]
- [ ] [depends:tenstorrent.harness.repeat_stats] L1 Add the long-run mode that records loss and gradient norm at a named cadence. r[tenstorrent.harness.long_run]
- [ ] [depends:tenstorrent.harness.long_run] L2 Add divergence and non-finite detection that stops or fails the run and names the step. r[tenstorrent.harness.long_run]

## Phase 2: Thermal telemetry

- [ ] [depends:tenstorrent.harness.long_run] T1 Add power and temperature telemetry samples at a named cadence over the long run. r[tenstorrent.harness.thermal]

## Phase 3: Held-out eval

- [ ] [depends:tenstorrent.harness.long_run] E1 Add the held-out eval split and the training-value metric, distinct from the periodic reference batch. r[tenstorrent.harness.heldout_eval]

## Phase 4: Verification and documentation

- [ ] [parallel] V1 Add positive fixtures for a stable long-run trajectory, a finite thermal series, and a held-out eval result. r[tenstorrent.harness.long_run] r[tenstorrent.harness.heldout_eval]
- [ ] [parallel] V2 Add negative fixtures for divergence, a non-finite thermal sample, and an eval claim on the reference batch only. r[tenstorrent.harness.thermal] r[tenstorrent.harness.heldout_eval]
- [ ] [serial] D1 Update each package README to report long-run and held-out results and revise the non-claims section. r[tenstorrent.harness.long_run]
- [ ] [depends:tenstorrent.harness.heldout_eval] G1 Run formatting, package tests, Nix checks, Cairn validation, and all lifecycle gates. r[tenstorrent.harness.heldout_eval]

## Verification Coverage

- `Scenario: Stable long run records trajectory` -> L1, V1
- `Scenario: Divergence is detected and named` -> L2, V2
- `Scenario: Thermal series is recorded and bound` -> T1, V1
- `Scenario: Non-finite thermal sample is rejected` -> T1, V2
- `Scenario: Held-out eval result is bound to its split` -> E1, V1
- `Scenario: Eval on the reference batch is rejected` -> E1, V2
