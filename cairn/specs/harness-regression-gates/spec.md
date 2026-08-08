# Harness Regression Gates Specification

## Purpose

Defines the `harness-regression-gates` capability.

## Requirements

### Requirement: Baseline store is checked in and keyed
r[tenstorrent.harness.baseline_store] The harness MUST keep a checked-in baseline store keyed by named profile and machine-facts identity. Each baseline SHALL bind its metrics, its run identity, and a BLAKE3 identity over the baseline record.

#### Scenario: Valid baseline is admitted
r[tenstorrent.harness.baseline_store.valid]
- GIVEN a baseline with a named profile, a machine-facts identity, and measured metrics
- WHEN baseline admission runs
- THEN the baseline SHALL bind one deterministic identity.

#### Scenario: Baseline with mismatched identity is rejected
r[tenstorrent.harness.baseline_store.invalid]
- GIVEN a baseline whose machine-facts identity or profile identity conflicts
- WHEN baseline admission runs
- THEN admission MUST fail and name the conflict.

### Requirement: Regression gate blocks on lost performance
r[tenstorrent.harness.regression_gate] A regression gate MUST fail a run when a measured metric regresses beyond the named bound against the matching baseline. The gate SHALL compare the same metric, on the same machine-facts identity, within the named tolerance.

#### Scenario: Metric within bound passes the gate
r[tenstorrent.harness.regression_gate.valid]
- GIVEN a measured metric that regresses less than the named bound
- WHEN the regression gate runs
- THEN the run passes and the receipt records the comparison.

#### Scenario: Metric beyond bound fails the gate
r[tenstorrent.harness.regression_gate.invalid]
- GIVEN a measured metric that regresses more than the named bound
- WHEN the regression gate runs
- THEN the run fails and the receipt names the regressed metric and its bound.

### Requirement: Determinism agreement is measured
r[tenstorrent.harness.determinism] A measurement run MUST compare two identical runs and report a determinism-agreement metric. The metric SHALL record the largest numerical difference and whether the runs agree within a named tolerance.

#### Scenario: Identical runs agree
r[tenstorrent.harness.determinism.valid]
- GIVEN two runs of the same step with identical inputs and profile
- WHEN the determinism check runs
- THEN the check SHALL report the agreement metric and its tolerance result.

#### Scenario: Divergent runs are surfaced
r[tenstorrent.harness.determinism.invalid]
- GIVEN two runs whose outputs differ more than the named tolerance
- WHEN the determinism check runs
- THEN the check MUST fail and report the largest difference.

### Requirement: Metric sanity fixtures gate the measured numbers
r[tenstorrent.harness.metric_sanity] Admitted metric data MUST pass positive and negative metric-sanity checks. A metric SHALL be non-negative, finite, and consistent with its repeat count. A non-finite, negative, or inconsistent metric MUST be rejected.

#### Scenario: Finite consistent metrics are admitted
r[tenstorrent.harness.metric_sanity.valid]
- GIVEN metrics that are finite, non-negative, and consistent with the repeat count
- WHEN metric sanity runs
- THEN admission SHALL pass and bind the metrics.

#### Scenario: Non-finite or inconsistent metrics are rejected
r[tenstorrent.harness.metric_sanity.invalid]
- GIVEN a metric that is non-finite, negative, or inconsistent with its repeat count
- WHEN metric sanity runs
- THEN admission MUST fail and name the offending metric.
