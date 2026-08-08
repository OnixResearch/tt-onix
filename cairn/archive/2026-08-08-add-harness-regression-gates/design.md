## Context

The measurement change records stage time, speedup, and machine facts. It does not
yet tell us how busy the device is or which operation dominates. A total step time
hides whether the harness is compute-bound, memory-bound, or launch-bound.

The regression-gate change stores baselines but measures only aggregate metrics. It
does not attribute time to an operation.

Device-utilization and op-level latency data turn "the step is slow" into "this
specific operation is the bottleneck". That attribute is what tells the author where
to optimize.

## Decisions

### Decision: Keep baselines keyed by machine-facts identity

**Choice:** Key each baseline by the profile and machine-facts identity so a metric
only compares against runs on the same board, firmware, PCIe, and mesh.

**Rationale:** A comparison across different firmware or boards is not meaningful. A
fail-closed key prevents false regressions and false passes.

### Decision: Reuse the existing admission core for gate decisions

**Choice:** Put the regression decision in the pure admission core. The CLI passes
the measured receipt and the baseline; the core returns pass or fail with the
regressed metric and its bound.

**Rationale:** The decision is deterministic and testable without a device. Keeping
it in core lets the gate run device-free on fixtures.

### Decision: Report determinism as a compared metric

**Choice:** Model determinism agreement as a metric: the largest numerical
difference between two runs and a pass-or-fail against the named tolerance.

**Rationale:** A determinism metric fits the existing receipt and regression shape
and can be gated like any other number.

### Decision: Make metric sanity a requirement on admission

**Choice:** Treat non-finite, negative, or repeat-inconsistent metrics as invalid
before the regression gate reads them.

**Rationale:** A regression gate that consumes a NaN or a negative time cannot give a
true verdict. Sanity admission keeps the gate honest.

## Validation Strategy

Device-free positive checks cover baseline admission, a passing gate, determinism
agreement, and metric sanity.

Device-free negative checks cover a baseline identity conflict, a gate failure, a
divergent determinism result, and a non-finite or inconsistent metric. Each negative
case must name the offending record.

Existing parity, measurement, and receipt checks keep their current results.

No validation in this change proves real long-run stability or training value.

## Risks / Trade-offs

- Baselines age as firmware and tooling change. The machine-facts key must be strict
  so an aged baseline is not silently reused.
- The determinism comparison doubles some run cost. It must reuse the repeated runs
  from the measurement change.
- A too-tight regression bound causes flaky gates; a too-loose bound lets regressions
  through. The bound must be named per metric in the profile.

## Non-Goals

- Long-run stability or thermal telemetry.
- Device utilization or op-level latency tables.
- Held-out eval loss or training-quality metrics.
- Scaling efficiency curves beyond the named profiles.
- A live metrics dashboard or external timeseries service.
