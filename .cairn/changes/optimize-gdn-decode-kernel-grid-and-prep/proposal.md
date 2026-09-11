## Why

GDN decode is 26.34 ms of the 56.33 ms P150x2 step, which is 47 percent. Per
layer that is 0.549 ms across 48 layers.

The upstream author of the fused GDN kernel published the remaining headroom in
priority order on issue 50475. The first two items are occupancy and host
preparation, and both are launch and scheduling costs rather than arithmetic:

- **Grid use.** The kernel opens one program instance per `(batch, value head)`.
  At TP=4 that is 8 instances against roughly 110 compute cores, so about 7
  percent of the grid is busy. The fix is to split the 128x128 recurrent state
  across cores per head, or to pipeline heads.
- **Host-side preparation.** Five fp32 typecasts, two L2 norms, and the query
  scale run as separate ops before the kernel. Folding them into the reader or
  the compute kernel removes that launch chain.

This repository measured the same shape of problem from the other side. The
hierarchical readback and component probes both failed their stability bounds
on host-side phases, and the decode-component evaluator rejected attribution
partly on host input-update instability.

The two remaining upstream items are out of scope here. The fp32-to-bf8 input
question is a precision change that needs its own accuracy gate. The verify
launch-bound item belongs to the fused verify path.

## What Changes

- Split the recurrent state across cores so a single value head no longer owns
  one whole core, and pipeline head groups where the state split alone is not
  enough.
- Fold the fp32 typecasts, the two L2 norms, and the query scale into the reader
  or compute kernel, removing them from the host op chain.
- Add a pure, device-free occupancy core that computes the per-core state split
  and the head-group schedule, and that rejects malformed plans.
- Gate on exact recurrence parity plus an occupancy floor and a step-time bound.

This change does not change the GDN numeric type, does not admit a new kernel
variant into production, and does not touch the verify path.

## Impact

- **Files**: `pkgs/qwen36/` (kernel source overlay, occupancy core, check
  probes), this change directory, and the physical evidence tree.
- **Testing**: qwen36 device-free check with positive and negative occupancy
  probes, package build, kernel hazard audits, a fresh Blackhole simulator
  smoke, and matched physical roots with exact recurrence parity.
- **Production**: unchanged. The candidate kernel stays behind its own gate and
  the admitted kernel remains the served route until the screen passes.
