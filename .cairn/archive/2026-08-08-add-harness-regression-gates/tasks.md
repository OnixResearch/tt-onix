# Tasks: Harness regression gates

## Phase 1: Baseline store

- [x] [serial] B1 Record baseline package tests and Cairn validation before edits. r[tenstorrent.harness.baseline_store]
- [x] [depends:tenstorrent.harness.repeat_stats] S1 Add the checked-in baseline store keyed by named profile and machine-facts identity. r[tenstorrent.harness.baseline_store]

## Phase 2: Regression gate and determinism

- [x] [depends:tenstorrent.harness.baseline_store] G1 Add the regression gate that fails on a metric regressing past the named bound against the matching baseline. r[tenstorrent.harness.regression_gate]
- [x] [depends:tenstorrent.harness.repeat_stats] D1 Add the determinism-agreement metric that compares two identical runs. r[tenstorrent.harness.determinism]

## Phase 3: Metric sanity

- [x] [depends:tenstorrent.harness.baseline_store] C1 Add metric-sanity checks for non-negative, finite, repeat-consistent metrics. r[tenstorrent.harness.metric_sanity]

## Phase 4: Verification and documentation

- [x] [parallel] V1 Add positive fixtures for baseline admission, gate pass, determinism agreement, and metric sanity. r[tenstorrent.harness.baseline_store] r[tenstorrent.harness.metric_sanity]
- [x] [parallel] V2 Add negative fixtures for baseline identity conflict, gate failure on regression, divergent runs, and non-finite or inconsistent metrics. r[tenstorrent.harness.regression_gate] r[tenstorrent.harness.determinism]
- [x] [serial] D2 Update each package README with a regression and determinism section. r[tenstorrent.harness.regression_gate]
- [x] [depends:tenstorrent.harness.regression_gate] G2 Run formatting, package tests, Nix checks, Cairn validation, and all lifecycle gates. r[tenstorrent.harness.regression_gate]

## Verification Coverage

- `Scenario: Valid baseline is admitted` -> S1, V1
- `Scenario: Baseline with mismatched identity is rejected` -> S1, V2
- `Scenario: Metric within bound passes the gate` -> G1, V1
- `Scenario: Metric beyond bound fails the gate` -> G1, V2
- `Scenario: Identical runs agree` -> D1, V1
- `Scenario: Divergent runs are surfaced` -> D1, V2
- `Scenario: Finite consistent metrics are admitted` -> C1, V1
- `Scenario: Non-finite or inconsistent metrics are rejected` -> C1, V2
