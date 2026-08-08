# Harness Performance Measurement Specification Delta

## ADDED Requirements

### Requirement: Measurement profile is distinct from parity profile
r[tenstorrent.harness.measurement_profile] A harness runner MUST expose a named measurement profile that is distinct from the parity profile. The measurement profile SHALL define the repeat count, the metric set, and the machine-facts set. It MUST NOT relax any parity tolerance.

#### Scenario: Measurement profile exports deterministic metrics
r[tenstorrent.harness.measurement_profile.valid]
- GIVEN a named measurement profile with a repeat count and a metric set
- WHEN profile export runs twice
- THEN both outputs and their BLAKE3 identities MUST match.

#### Scenario: Measurement profile cannot gate parity
r[tenstorrent.harness.measurement_profile.scope]
- GIVEN a measurement profile that attempts to change a parity tolerance
- WHEN the harness validates the profile
- THEN validation MUST fail before any device access.

### Requirement: Receipt binds stage time breakdown
r[tenstorrent.harness.stage_breakdown] A measurement receipt MUST bind a wall-time breakdown for each named stage. The stages MUST cover launch, compile, probe or step, checkpoint, and teardown. Each stage time MUST be non-negative and finite.

#### Scenario: Stage times are recorded and bounded
r[tenstorrent.harness.stage_breakdown.valid]
- GIVEN a completed measurement run with a named profile
- WHEN the harness writes the receipt
- THEN the receipt SHALL contain one non-negative finite time per stage.

#### Scenario: Negative or non-finite stage time is rejected
r[tenstorrent.harness.stage_breakdown.invalid]
- GIVEN a measured stage time that is negative or non-finite
- WHEN the admission core validates the receipt
- THEN admission MUST fail and name the invalid stage.

### Requirement: Receipt binds CPU-baseline speedup ratio
r[tenstorrent.harness.speedup_ratio] A measurement receipt MUST bind a CPU-baseline speedup ratio. The ratio SHALL use the stage time from a measured CPU reference under the same harness. The receipt MUST name the exact CPU run it compares against.

#### Scenario: Speedup ratio is computed against a bound baseline
r[tenstorrent.harness.speedup_ratio.valid]
- GIVEN a measured device time and a measured CPU reference time for the same step
- WHEN the harness computes the ratio
- THEN the receipt SHALL bind the ratio and the CPU baseline identity.

#### Scenario: Speedup ratio without a CPU baseline is rejected
r[tenstorrent.harness.speedup_ratio.invalid]
- GIVEN a measurement run with no bound CPU reference time
- WHEN admission runs
- THEN the receipt MUST NOT report a speedup ratio.

### Requirement: Receipt binds machine facts
r[tenstorrent.harness.machine_facts] A measurement receipt MUST bind machine facts for the run. The facts SHALL cover board identity, firmware version, PCIe link, power state, and mesh descriptor identity. The receipt MUST reject a run whose machine facts cannot be read.

#### Scenario: Machine facts are bound to the receipt
r[tenstorrent.harness.machine_facts.valid]
- GIVEN a run that reads the board, firmware, PCIe, power, and mesh facts
- WHEN the harness writes the receipt
- THEN each fact SHALL bind to the receipt with its BLAKE3 identity.

#### Scenario: Missing machine facts fail admission
r[tenstorrent.harness.machine_facts.invalid]
- GIVEN a run with one or more unreadable machine facts
- WHEN admission runs
- THEN the receipt MUST be rejected and MUST name the missing fact.

### Requirement: Measurement run repeats and reports distribution
r[tenstorrent.harness.repeat_stats] A measurement run MUST repeat the step M times per the named profile and report mean, p50, p99, and standard deviation for time and for loss. M MUST be at least three unless the profile records a lower bound.

#### Scenario: Repeated run reports a distribution
r[tenstorrent.harness.repeat_stats.valid]
- GIVEN a measurement run that executes the step M times with M at least three
- WHEN the harness summarizes the runs
- THEN the receipt SHALL contain mean, p50, p99, and standard deviation for time and loss.

#### Scenario: Repeats below the bound are rejected
r[tenstorrent.harness.repeat_stats.invalid]
- GIVEN a measurement run whose repeat count is below the profile bound
- WHEN the admission core validates the run
- THEN admission MUST fail.
