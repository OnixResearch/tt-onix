## Context

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

## Decisions

### Decision: Add a measurement profile distinct from parity

**Choice:** Add a named measurement profile that keeps the parity tolerances and
adds a repeat count, a metric set, and a machine-facts set.

**Rationale:** One profile cannot gate both correctness and value without confusing
their contracts. A measurement profile must never relax a parity tolerance.

### Decision: Measure wall time per stage

**Choice:** Record the wall time for launch, compile, probe or step, checkpoint, and
teardown as separate receipt fields. The sum of the stages is the end-to-end time.

**Rationale:** The step's total obscures where the time goes. In many setups launch
and compile dominate compute. A speedup claim is weak until the fixed overhead is
subtracted.

### Decision: Bind a CPU-baseline speedup ratio

**Choice:** Run the same step on the CPU reference under the same harness and compute
the device-to-CPU speedup ratio as a receipt field. The receipt names the exact CPU
run it compares against.

**Rationale:** A bare tokens-per-second number is not verifiable. A ratio with a
bound baseline identity is reproducible and attributable.

### Decision: Bind machine facts to every receipt

**Choice:** Read board, firmware, PCIe link, power state, and mesh descriptor
identity and bind each with BLAKE3. Reject a run whose machine facts cannot be read.

**Rationale:** A performance number is meaningless unless it is comparable across
cards and firmware. Machine facts make the number attributable and reproducible.

### Decision: Repeat the step M times

**Choice:** Repeat the step M times per the profile and report mean, p50, p99, and
standard deviation for time and loss. M is at least three unless the profile lowers
it.

**Rationale:** One receipt is luck. A distribution catches flakiness and
nondeterminism, which are the classes of bugs this repository already chases.

## Validation Strategy

Device-free positive checks cover the measurement profile export, the stage
breakdown, the speedup ratio, the machine-facts binding, and the repeat statistics.

Device-free negative checks cover a parity-relaxing profile, a non-finite stage
time, a missing CPU baseline, missing machine facts, and a low repeat count. Each
negative case must fail cleanly and name the offending field.

Existing parity, profile, and receipt checks must keep their current results.

No validation in this change can prove real-world speedup, long-run stability, or
training value.

## Risks / Trade-offs

- Machine facts depend on tools that can be absent on some hosts. The harness must
  fail closed and name the missing fact.
- A CPU reference can be slow to run. The change must reuse the existing reference
  runner so the added cost is bounded.
- Repeating steps M times multiplies device time by M. The profile must use the
  smallest M that still yields a stable distribution.

## Non-Goals

- Long-run stability or thermal telemetry.
- Held-out eval loss or training-quality metrics.
- Device utilization or op-level latency tables.
- A regression-gate layer against stored baselines.
- Scaling efficiency curves beyond the named profiles.
