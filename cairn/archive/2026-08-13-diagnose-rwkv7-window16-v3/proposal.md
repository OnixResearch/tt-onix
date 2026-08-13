## Why

Corrected candidate v2 first differs from the pinned CPU continuation at generated-token index `10`. Normal dispatch and persistent trace agree, so trace replay is not the cause.

A new candidate cannot reuse v2 evidence. A bounded diagnosis must find the first full-model operation that diverges from the CPU authority.

## What Changes

- Add a fail-closed diagnostic mode for the known failing prefix.
- Compare the fused path with the package-matched generic TT-NN path at bounded layer and operation boundaries.
- Retain aggregate hashes, maximum errors, and the first divergence only.
- Create a new immutable candidate only after the diagnosis proves and corrects one root cause.
- Keep production windows `[2, 4, 8]` until a new candidate passes every admission gate.

## Impact

- **Files**: RWKV-7 runtime diagnostics, evidence validation, Nix package wiring, Cairn evidence, and the accepted runtime specification
- **Testing**: positive and negative unit tests, package checks, source and closure checks, bounded simulator checks, and one bounded physical diagnosis
