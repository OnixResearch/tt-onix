## Why

Every verified receipt in the repository proves correctness, not value. Probes show
that a GPU step matches the CPU authority inside named tolerances. They do not show
how long the step took, where the time went, or whether the hardware is faster than
the reference.

The repository documents speedup, scaling efficiency, and long-run stability as
explicit non-claims. These gaps exist because the harness has no measurement phase.
It only has a parity phase.

A measurement phase changes the question from "is it correct?" to "is it worth
running?". We cannot make a speedup or stability claim without a repeatable,
attributable, and regression-checkable performance number.

## What Changes

- Add a named measurement profile that is distinct from the parity profile.
- Add a stage time breakdown over launch, compile, probe or step, checkpoint, and
  teardown.
- Add a measured CPU-baseline speedup ratio as an explicit receipt field.
- Add machine facts (board, firmware, PCIe link, power state, mesh) to every receipt.
- Add a perf run mode that repeats a step M times and records mean, p50, p99, and
  standard deviation.
- Add positive and negative metric-sanity fixtures.

## Impact

- **Packages**: `rwkv7-lora-p150a`, `rwkv7-lora-dual-p150a`, and the shared profile
  core gain a measurement phase and machine-facts binding.
- **Receipts**: each receipt binds machine facts and measured metrics with BLAKE3.
- **Validation**: metric-sanity checks validate the measured numbers device-free.
- **Documentation**: each package README gains a measurement section and a revised
  non-claims section.

## Out of Scope

- Long-run stability, thermal telemetry, or held-out eval loss.
- Device utilization or op-level latency tables.
- A regression-gate layer against stored baselines.
- Scaling efficiency curves beyond the named profiles.
- Any change to parity or correctness admission semantics.
