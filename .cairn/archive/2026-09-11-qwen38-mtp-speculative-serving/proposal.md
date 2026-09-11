# Gated MTP speculative serving

## Why

The BF16 MTP-head pilot recorded mean acceptance 2.4 and diagnostic emitted-token speedup 1.382x at draft count 3. Its timing-spread gate still failed. It did not prove sustained serving safety or performance.

Source revision `51e07b8079b1b91e7c64fbf15c692d0df98fcc35` now contains serving wiring, a feedback pool, and stage watchdogs. The old premise that no integration exists is stale.

Finding 1 in `docs/research/qwen38-p150x2-2026-09-04.md` identifies remaining lifecycle defects. Ruff confirms an undefined timer variable. Source review also finds ineffective loss accounting, stale prefix-zero feedback, lazy allocation after capture, and missing shared device ownership.

## What Changes

- Complete the existing greedy MTP path rather than create a second implementation.
- Preserve the exact seed, draft, continuation, and hybrid-state contract across multiple rounds.
- Separate pure policy from watchdog threads, clocks, and device effects.
- Make timeout quarantine, request deadlines, cleanup, and fallback explicit.
- Require a fresh, complete serving screen before owner enablement.

## Impact

The existing round-policy core, verifier, drafter, trace adapter, server, and check suite own this work. The gate remains default-off. Sampled and batch behavior must remain unchanged on the off route.

This research update changes documents only. Existing completed task markers remain historical records. New tests and physical admission remain open. No failed receipt becomes passing evidence through this update.
