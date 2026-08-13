# Evaluate corrected RWKV-7 persistent window 16

## Why

The rejected v1 candidate failed exact CPU continuation parity because its fused WKV path used the wrong recurrence and readout contracts. The diagnostic split now matches package-matched generic TT-NN exactly on physical Blackhole hardware. A new immutable candidate must repeat every admission gate before production can change.

## What Changes

- Add a typed v2 candidate profile without modifying the rejected v1 identity.
- Build a v2 runtime and evidence closure with exact HiFi4 recurrence and separate HiFi2 BF16 readout.
- Repeat simulator, physical correctness, lifecycle, profiler, A/B, canary, monitoring, deployment, and observation gates.
- Admit window `16` only if every gate passes. Retain `[2, 4, 8]` after any valid negative result.

## Impact

- **Files**: candidate profile, contracts, package wiring, evidence policies, Cairn artifacts, and accepted production profile only after all gates pass.
- **Tests**: positive and negative profile tests, packages, closures, simulator smoke, physical parity, lifecycle, profiler, A/B, canary, monitoring, deployment, observation, quality, and Cairn gates.

## Non-goals

- This change does not reuse or mutate candidate v1.
- Simulator evidence does not prove physical placement, performance, determinism, or admission.
- Diagnostic parity does not bypass candidate correctness or operational gates.
