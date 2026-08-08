## Why

The repository's bounded runs prove that a few steps are correct. They do not prove
that the hardware stays correct and stable over an actual fine-tuning run that lasts
hours.

The bounded multi-step run drove loss to 0.0 against a periodic reference batch.
That proves the hardware overfits the reference batch. It does not prove the model
learned anything useful.

Long-run stability and training value are the two non-claims that decide whether the
hardware is worth using for real training. Neither can be measured from an 8-step
run.

## What Changes

- Add a long-run measurement mode that records loss and gradient norm over many
  steps and detects divergence or non-finite values.
- Add power and temperature telemetry over time during a long run.
- Add a held-out eval loss metric on a real eval split, distinct from the periodic
  reference batch.
- Add positive and negative fixtures for divergence detection, thermal bounds, and
  held-out eval admission.

## Impact

- **Packages**: the shared harness gains a long-run mode and a held-out eval metric.
- **Receipts**: each receipt binds the long-run trajectory, thermal telemetry, and
  held-out eval result with BLAKE3.
- **Validation**: divergence detection and held-out eval fixtures run device-free.
- **Documentation**: each package README updates its non-claims section for stability
  and training value.

## Out of Scope

- Device utilization or op-level latency tables.
- A regression-gate layer against stored baselines.
- Scaling efficiency curves beyond the named profiles.
- A live metrics dashboard.
- Any change to the correctness admission semantics of a single step.

## Dependencies

- This change builds on the measurement-profile change for the run shape and repeats.
- It does not require the regression-gate or device-utilization changes.
