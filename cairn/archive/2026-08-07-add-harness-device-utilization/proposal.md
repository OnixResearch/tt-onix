## Why

The measurement change records total step time and speedup. It does not tell us why
a step is slow. A total obscures the difference between a compute-bound, a
memory-bound, and a launch-bound run.

The repository's correctness proofs show that kernels execute. They do not show
whether the compute pipeline is busy or idle. A matmul-heavy model can still run at
low device utilization if the harness is memory-bound or serialized by the
recurrence.

Op-level latency data turns "the step is slow" into "this specific operation is the
bottleneck". Without it, optimization work guesses instead of measures.

## What Changes

- Add device-utilization metrics: math or ALU busy fraction and NoC idle fraction.
- Add a memory-pressure metric: peak L1 or SRAM and DRAM or sysmem per device.
- Add a per-operation latency table with mean and p99 latency per TT-NN operation.
- Add a scaling-efficiency metric that compares one-device and two-device time per
  step against the ideal data-parallel line.
- Add positive and negative fixtures for each new metric.

## Impact

- **Packages**: the shared harness gains utilization, memory, op-latency, and scaling
  metrics in the measurement phase.
- **Receipts**: each receipt binds utilization, memory, op-latency, and scaling facts
  with BLAKE3.
- **Validation**: utilization and op-latency fixtures run device-free.
- **Documentation**: each package README gains a utilization and bottleneck section.

## Out of Scope

- Long-run stability, thermal telemetry, or power telemetry.
- Held-out eval loss or training-quality metrics.
- A regression-gate layer against stored baselines (covered by a sibling change).
- A live metrics dashboard.

## Dependencies

- This change builds on the measurement profile and repeat-statistics change
  (`add-harness-perf-measurement`), which supplies the measured run to instrument.
- It does not require the regression-gate change, but regression gates can reuse the
  op-latency table for attribution.
