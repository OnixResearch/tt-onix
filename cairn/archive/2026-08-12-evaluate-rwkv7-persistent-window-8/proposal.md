# Evaluate RWKV-7 persistent decode window 8

## Why

The accepted profile admits persistent decode windows `2` and `4`. The deployed observation was clean, but it gives no evidence for window `8`.

Window `8` increases trace storage, retained tensor lifetime, cancellation delay, and replay work. It needs separate simulator, parity, profiler, physical A/B, and canary evidence.

## What Changes

- Add a fail-closed window-`8` evaluation profile while the accepted production profile remains at `[2, 4]`.
- Extend runtime, evidence, monitoring, and canary contracts for a bounded eight-step candidate.
- Run simulator smoke, terminal lifecycle probes, trace profiling, order-balanced physical A/B measurements, and one bounded canary.
- Admit window `8` only after every required gate passes. Retain `[2, 4]` if any gate fails.

## Impact

- **Files**: `pkgs/rwkv7-p150x2`, `pkgs/rwkv7-p150x2-evidence`, `flake-modules/onix-packages.nix`, and Cairn evidence
- **Testing**: runtime and evidence package checks, simulator smoke, physical lifecycle probes, profiler validation, one-card and two-card A/B, canary monitoring, repository quality, production closure, and Cairn gates
- **Operator boundary**: no daemon, live routing, unbounded device loop, or automatic retry is added
