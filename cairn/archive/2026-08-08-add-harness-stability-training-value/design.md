## Context

The repository's bounded runs prove that a few steps are correct. They do not prove
that the hardware stays correct and stable over a real fine-tuning run that lasts
hours.

The bounded multi-step run drove loss to 0.0 against a periodic reference batch.
That proves the hardware overfits the reference batch. It does not prove the model
learned anything useful.

Long-run stability and training value are the two non-claims that decide whether the
hardware is worth using for real training. Neither can be measured from an 8-step
run.

## Decisions

### Decision: Add a long-run mode with divergence detection

**Choice:** Add a long-run measurement mode that records loss and gradient norm at a
named cadence and stops or fails the run on divergence or a non-finite value.

**Rationale:** A run that goes non-finite partway through is unstable. The run must
stop at the failing step and name it so the failure is attributable and reproducible.

### Decision: Record thermal telemetry as a bound series

**Choice:** Record power and temperature at a named cadence as a finite series bound
to the device identity.

**Rationale:** Stability over hours includes thermal endurance. A finite, bound
series is the evidence that the hardware thermals stayed inside limits.

### Decision: Keep held-out eval distinct from the reference batch

**Choice:** Report training value on a held-out eval split distinct from the periodic
reference batch, and bind the split identity in the receipt.

**Rationale:** A metric computed only on the reference batch measures fitting, not
learning. A held-out split is the difference between "loss went down" and "the model
learned".

## Validation Strategy

Device-free positive checks cover a stable long-run trajectory, a finite thermal
series, and a held-out eval result bound to its split.

Device-free negative checks cover divergence, a non-finite thermal sample, and an
eval claim computed on the reference batch only. Each negative case names the
offending step, sample, or split.

Existing parity, measurement, and receipt checks keep their current results.

No validation in this change proves downstream task performance or real-world
generalization.

## Risks / Trade-offs

- Thermal telemetry may not be exposed by every host tool. The run must record an
  unavailable source as a named gap, not a silent constant.
- A long run is expensive. The cadence and step budget must come from the profile so
  the cost is bounded and explicit.
- Held-out eval needs a real eval split. The split identity must be pinned so the
  result is reproducible.

## Non-Goals

- Device utilization or op-level latency tables.
- A regression-gate layer against stored baselines.
- Scaling efficiency curves beyond the named profiles.
- A live metrics dashboard.
- Any change to the correctness admission of a single step.
