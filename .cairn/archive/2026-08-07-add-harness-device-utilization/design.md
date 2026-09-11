## Context

The measurement change records total step time and speedup. It does not tell us why
a step is slow. A total obscures whether the run is compute-bound, memory-bound, or
launch-bound.

The correctness proofs in this repository show that kernels execute. They do not
show whether the compute pipeline stays busy or idles between operations.

Op-level latency attributes time to a specific operation. Utilization and memory
metrics say how hard the device is working. Together they tell the author where the
bottleneck is and whether the device has headroom to grow the batch or sequence.

## Decisions

### Decision: Instrument the existing measured run

**Choice:** Add utilization, memory, op-latency, and scaling collection inside the
measurement run from the measurement-profile change. Reuse the same profile and
repeat count.

**Rationale:** Instrumenting an already-measured run avoids a second, separate run
and keeps utilization consistent with the step time it explains.

### Decision: Name the source of each utilization metric

**Choice:** Record the source of each math busy and NoC idle fraction so the number
is attributable (device profiler, kernel counter, or simulator).

**Rationale:** Different tools expose different definitions of busy. Naming the
source keeps two runs comparable.

### Decision: Require the op table to cover the whole step

**Choice:** Treat a step that executes an operation absent from the latency table as
admission failure.

**Rationale:** A partial table hides the very operation that is the bottleneck. A
complete table is required for the "where to optimize" answer to be sound.

### Decision: Gate scaling claims on a real two-device run

**Choice:** Report a scaling ratio only when a two-device measurement exists for the
named profile. Reject a scaling claim that has only a one-device number.

**Rationale:** A scaling-efficiency number without the second device is speculation.
The fail-closed rule keeps the claim attributable.

## Validation Strategy

Device-free positive checks cover utilization bounds, memory against capacity, a
complete op-latency table, and a scaling ratio with a two-device run.

Device-free negative checks cover out-of-range utilization, an invalid memory peak,
a missing operation, and a scaling claim without a two-device run. Each negative
case names the offending field.

Existing parity, measurement, and receipt checks keep their current results.

No validation in this change proves thermal endurance or long-run stability.

## Risks / Trade-offs

- The device profiler may not be available on every host. The run must record an
  unavailable source as a named gap, not a silent zero.
- Op-level instrumentation adds overhead to the measured step. The overhead must be
  excluded from the step time it reports.
- A two-device run doubles hardware time. Scaling measurement must be an explicit
  opt-in for profiles that name both meshes.

## Non-Goals

- Long-run stability, thermal telemetry, or power telemetry.
- Held-out eval loss or training-quality metrics.
- A regression-gate layer against stored baselines, which is a sibling change.
- A live metrics dashboard.
