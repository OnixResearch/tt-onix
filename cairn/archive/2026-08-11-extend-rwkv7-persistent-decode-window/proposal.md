## Why

The admitted persistent decode path accepts only one fixed two-token window. This bound proves the mechanism, but it does not cover a useful bounded continuation or the retained conversational prompt fixture.

The next admission must preserve the existing two-token path while it adds a longer profile-owned window. It must reject unsupported lengths and retain normal dispatch without weakening parity, cleanup, profiler, or physical-gain gates.

## What Changes

- Add a typed set of admitted persistent decode windows to the Nickel profile.
- Derive request-local lifecycle bounds from an admitted window without changing the base profile identity.
- Admit a longer bounded trace only after exact token and FP32 recurrent-state parity.
- Add the retained `Hello` chat prompt and its pinned FP32 CPU-reference tokens to the persistent workload.
- Add cancellation-boundary, timeout, and repeated slot-recycling evidence.
- Repeat order-balanced one-card and two-card physical measurements before production admission.

## Impact

- **Files**: `pkgs/rwkv7-p150x2`, `pkgs/rwkv7-p150x2-evidence`, the accepted RWKV-7 runtime spec, and retained Cairn evidence.
- **Testing**: lifecycle unit tests, invalid Nickel fixtures, runtime self-tests, simulator smoke, physical parity, trace-only profiler evidence, order-balanced A/B runs, repository quality, and Cairn gates.
