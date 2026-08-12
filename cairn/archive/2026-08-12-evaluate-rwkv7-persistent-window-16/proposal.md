# Evaluate RWKV-7 persistent decode window 16

## Why

The accepted profile admits persistent decode windows `2`, `4`, and `8`. It gives no evidence for window `16`.

Window `16` increases trace storage, retained tensor lifetime, cancellation delay, and replay work. It needs separate simulator, parity, profiler, physical A/B, and canary evidence.

## What Changes

- Add a fail-closed window-`16` evaluation profile while production remains at `[2, 4, 8]`.
- Extend runtime, evidence, monitoring, and canary contracts for a bounded sixteen-step candidate.
- Run simulator smoke, lifecycle probes, trace profiling, physical A/B measurements, and one bounded canary.
- Admit window `16` only after every required gate passes. Retain `[2, 4, 8]` if a gate fails.

## Impact

- **Files**: `pkgs/rwkv7-p150x2`, `pkgs/rwkv7-p150x2-evidence`, `flake-modules/onix-packages.nix`, and Cairn evidence
- **Tests**: package checks, simulator smoke, lifecycle probes, profiler validation, physical A/B, canary monitoring, repository quality, and Cairn gates
- **Boundary**: this change adds no daemon, live routing, unbounded device loop, or automatic retry
