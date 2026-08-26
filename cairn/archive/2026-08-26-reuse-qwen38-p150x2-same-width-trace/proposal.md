## Why

Each served batch releases and recaptures its exact-width decode trace after prefill. Warm trace preparation costs about `0.64` seconds per request group. The pinned source supports prewarmed slot prefill, but earlier two-trace tests corrupted GDN buffers.

## What Changes

- Add a pure policy for preserve, release, reuse, and capture decisions.
- Keep at most one resident decode trace.
- Reuse the trace only for the same width after a complete reset and stable binding.
- Release the trace before prefill for width changes, binding drift, reset failure, or disabled policy.
- Add a closed Nickel profile and operator-only physical evidence mode.
- Record reuse, resident-trace count, capture time, and request wall time.
- Run matched `recapture,reuse,reuse,recapture` roots on physical devices `0` and `1`.
- Enable production reuse only if every correctness, identity, stability, and latency gate passes.
- Preserve the current recapture path and an exact rejected result if any gate fails.

## Impact

- **Files**: `pkgs/qwen36`, one Rust receipt script, the Qwen runtime spec, and physical evidence.
- **Testing**: Package checks, positive and negative policy tests, receipt self-tests, Cairn gates, matched physical roots, and production restoration.
