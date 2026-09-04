## Context

The service calls `prefill_paged_slots`, which retains sequential per-user prefill. The adopted `Qwen36Model.prefill_paged_grouped` already groups GDN and MLP work. Its default group cap assumes TP4 head geometry.

## Decisions

A pure planner consumes the runtime capability record, local heads, worker grid, bucket length, and scratch budget. It derives a group bound. At TP2, local value heads equal 24 rather than 12.

The old sequential kernel's stated head-work limit cannot justify a group of two TP2 users. The fused candidate needs its own exact bound. Do not turn the source's literal cap of eight into authority.

Reuse the grouped model path for eligible groups. Keep full-attention prefill per-user initially. Preserve actual lengths and slot destinations explicitly. Grouped projection and MLP work remain separate from recurrent state updates.

The control is the current device-staged per-user path, not the older host-staging path. A group-size fallback must occur before allocation.

## Validation

A device-free shape matrix covers short full buckets, masked tails, unequal lengths, empty groups, wrong TP geometry, and excessive scratch memory. State tests cover distinct prompts, reordered users, partial installation, and repeated slot use.

After runtime capability proof, run a representative simulator path. Then use an approved paired physical screen at B4. Rescreen B8 only after B4 passes on the exact candidate identity.

Retain B8's aggregate speedup floor 6.0, retention floor 0.75, TTFT cap 9.0, and tail cap 1.5. These are existing policy values, not new estimates. Faster prefill does not waive decode-retention failure.

## Alternatives and rollback

A fresh all-model batched forward duplicates existing source and increases the correctness surface. Blind full-bucket padding corrupts GDN carry. Both alternatives are rejected.

The current per-user path remains the fallback. A blocked or rejected B8 result closes the experiment without changing its default or thresholds.
