## Why

The coordinator batches equal generation bounds and waits for each complete group. It does not admit new work into a finished slot during decode.

Finding 5 in `docs/research/qwen38-p150x2-2026-09-04.md` identifies this limit. The rejected trace-reuse result also shows that fewer captures can increase request latency.

## What Changes

- Add a bounded slot state machine for staggered arrivals and unequal generation lengths.
- Make one worker own every device effect across greedy, sampled, and MTP routes.
- Evaluate trace-safe prefill and slot refill before any persistent scheduling admission.
- Start with B4 capacity and retain the current group coordinator as fallback.

## Impact

The scheduler core owns policy and transitions. The worker shell owns queue access, time, TTNN effects, and responses. Slot generations protect against stale completion events.

This change does not introduce a serving framework, multiple model copies, cross-tenant sharing, or automatic B8 admission. A source-level batch bucket is not a continuous scheduler.
