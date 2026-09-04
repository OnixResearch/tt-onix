## Why

The custom service still performs per-user model prefill. The adopted model already contains grouped prefill, but its group cap assumes TP4 geometry. TP2 doubles local GDN value heads.

The earlier B8 screen failed TTFT and decode-retention gates. Device staging changed the prefill baseline after that screen.

## What Changes

- Derive grouped-prefill eligibility from the exact runtime, local heads, bucket, and memory budget.
- Reuse `prefill_paged_grouped` where its contract matches.
- Preserve per-user full attention, valid-length masks, and state ownership.
- Compare the current staged path with grouped prefill at B4, then rescreen B8 without changing its gates.

## Impact

The pure planner belongs in `pkgs/qwen36/service/qwen36_service`. TTNN state installation remains in the adapter. Finding 4 in the September research report supplies the source basis.

B8 remains unadmitted until every gate passes. No larger context, default batch width, or service interruption is part of this proposal.
