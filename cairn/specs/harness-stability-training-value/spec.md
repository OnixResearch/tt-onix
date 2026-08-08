# Harness Stability Training Value Specification

## Purpose

Defines the `harness-stability-training-value` capability.

## Requirements

### Requirement: Long-run mode records trajectory and detects divergence
r[tenstorrent.harness.long_run] A long-run measurement mode MUST record loss and gradient norm at a named cadence over many steps. It MUST detect and name any divergence, non-finite value, or instability the moment it occurs.

#### Scenario: Stable long run records trajectory
r[tenstorrent.harness.long_run.valid]
- GIVEN a long-run mode with a cadence and a finite step budget
- WHEN the run completes without divergence
- THEN the receipt SHALL contain the full loss and gradient-norm trajectory.

#### Scenario: Divergence is detected and named
r[tenstorrent.harness.long_run.invalid]
- GIVEN a long run in which loss or gradient norm becomes non-finite or diverges
- WHEN the run observes the condition
- THEN the run MUST stop or mark the receipt failed and name the step and condition.

### Requirement: Thermal telemetry is bound over time
r[tenstorrent.harness.thermal] A long-run measurement mode MUST record power and temperature telemetry at a named cadence. Each sample SHALL be finite and the receipt SHALL bind the full thermal series and its device identity.

#### Scenario: Thermal series is recorded and bound
r[tenstorrent.harness.thermal.valid]
- GIVEN a long-run mode with thermal telemetry available
- WHEN the run records samples
- THEN the receipt SHALL bind the finite thermal series and the device identity.

#### Scenario: Non-finite thermal sample is rejected
r[tenstorrent.harness.thermal.invalid]
- GIVEN a thermal sample that is non-finite
- WHEN the harness writes the receipt
- THEN the sample or the run MUST be marked invalid and the device MUST be named.

### Requirement: Held-out eval is distinct from the reference batch
r[tenstorrent.harness.heldout_eval] A training-value metric MUST report loss or a quality metric on a held-out eval split distinct from the periodic reference batch. The receipt MUST bind the eval split identity and MUST NOT claim generalization beyond it.

#### Scenario: Held-out eval result is bound to its split
r[tenstorrent.harness.heldout_eval.valid]
- GIVEN a held-out eval split distinct from the periodic reference batch
- WHEN the training-value check runs
- THEN the receipt SHALL bind the eval result and the split identity.

#### Scenario: Eval on the reference batch is rejected
r[tenstorrent.harness.heldout_eval.invalid]
- GIVEN a training-value metric computed on the periodic reference batch only
- WHEN the admission core validates the claim
- THEN the claim MUST be rejected as not held out.
