## Why

The measurement change makes the harness record performance and machine facts. It
does not yet make those numbers useful over time. A one-off speedup receipt says
nothing about whether the next change made the harness faster or slower.

Without a regression gate, a performance regression can ship silently. The harness
records numbers but cannot act on them.

A regression-gate layer stores measured baselines and fails a run when a metric
regresses beyond the named bound. This turns metrics into gates and makes them a
living part of the lifecycle instead of a one-time receipt.

## What Changes

- Add a checked-in baseline store keyed by named profile and machine-facts identity.
- Add a regression gate that fails when a measured metric regresses past the named
  bound.
- Add a determinism-agreement metric that compares two identical runs.
- Add positive and negative fixtures for baseline admission, regression failure,
  determinism, and metric sanity.

## Impact

- **Packages**: the shared harness gains a baseline store and a regression gate.
- **Receipts**: each receipt binds the baseline identity it is compared against.
- **Validation**: regression and determinism checks run device-free on fixtures.
- **Documentation**: each package README gains a regression section.

## Out of Scope

- Long-run stability or thermal telemetry.
- Device utilization or op-level latency tables.
- Held-out eval loss or training-quality metrics.
- Scaling efficiency curves beyond the named profiles.
- A live metrics dashboard or external timeseries service.
