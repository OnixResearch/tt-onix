## Why

Persistent decode now has physical production admission for windows `2` and `4`. Its worker receipt contains detailed evidence, but operators lack one stable, low-cardinality telemetry event for admission, fallback, timing, parity, and cleanup.

Without a versioned telemetry contract, production monitoring must depend on internal receipt fields. This creates schema drift risk and makes safe canary aggregation difficult.

## What Changes

- Add a schema-versioned `persistent_decode_telemetry` event to every successful fast-worker receipt.
- Record eligibility, selected mode and window, fallback reason, terminal outcome counts, trace timings, parity status, and cleanup status.
- Keep telemetry aggregate-only. Exclude prompt text, token IDs, binding identities, filesystem paths, and request identifiers.
- Add a pure telemetry summary core with positive and negative C++ tests.
- Add strict Rust validation that cross-checks telemetry against the detailed worker receipt.
- Retain fail-closed normal dispatch and all existing admission rules.

## Impact

- **Files**: `pkgs/rwkv7-p150x2/src`, `pkgs/rwkv7-p150x2-evidence/src`, package documentation, and Cairn runtime specifications.
- **Testing**: Run the runtime self-test, Rust positive and negative tests, receipt validation, formatting, source-boundary, closure, and repository-quality gates.
