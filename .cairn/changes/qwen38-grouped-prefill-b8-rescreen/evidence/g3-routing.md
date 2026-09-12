# g3/g4 (device-free half): the adapter consults the planner, gated and fail-closed

Recorded 2026-09-12 for change `qwen38-grouped-prefill-b8-rescreen`.

## What was wired

`BatchedTracedDecodeAdapter.plan_prefill_route(width, valid_lengths)` now derives
the route before any device call:

- reads the package's own capability record with
  `load_runtime_capability_record()` (missing, malformed, mis-versioned, or
  non-boolean → one-instance-per-core, never a chunked claim);
- measures the grid with `_compute_cores()` and refuses grouping when the grid
  cannot be read (`compute-grid-unavailable`) rather than guessing a core count;
- reads `local_value_heads` from the model's GDN layers and the bucket chunk from
  `args.gdn_chunk_size` (128);
- calls the pure planner with `capacity_units = maximum_width` rows and
  `units_per_user = 1`, so the memory bound is expressed in the unit the adapter
  actually has (rows), while the fixtures use bytes;
- re-checks that the plan's destinations are exactly the user rows in order.
  Any other mapping returns the per-user route with
  `non-positional-state-destinations`, because the grouped path installs state by
  row (`row u = token_ids_list[u]`, per `_assemble_groups_gdn_dev`).

When the route is grouped, `prefill_users` calls
`model.prefill_paged_grouped(..., group_size=plan.group_size)`. The model still
applies its own defensive `min(group_size, gdn_max_bg)` ceiling, so the planner's
tighter derived bound always wins (TP2 one-instance → 4). First-token readiness
is observed from the returned host logits, and `BatchPrefillResult` now carries
`prefill_route` and `prefill_route_reason` so a receipt can show which route ran.

## The gate

Grouping is **opt-in** via `QWEN35_GROUPED_PREFILL` (`1/true/yes/on`), default
off. Reason: wiring changes the service source, which is an execution-identity
input, and the default path must keep its own evidence. With the gate off the
served route is byte-for-byte the previously measured one.

## Tests

`probe_grouped_prefill_routing` (device-free, runs in the package gate):

| Case | Expectation |
| --- | --- |
| gate unset / `0` / `off` / `maybe` | per-user, `grouped-prefill-disabled` |
| gate `1/true/yes/on/TRUE` | enabled |
| chunked capability, 8 users | grouped, group size 8 |
| one-instance capability, 8 users | grouped, group size 4 |
| unreadable grid | per-user, `compute-grid-unavailable` |
| bucket above the single-pass multiple | per-user, `bucket-above-single-pass` |
| missing / malformed / mis-versioned / non-boolean capability record | one-instance fallback |
| valid capability record | chunked claim read, source `package-runtime-capability` |
| `CHUNKED_VALIDATED_INSTANCES_PER_CORE` | must stay 2 without new evidence |

Package gate: `PASS grouped prefill eligibility`, `PASS grouped prefill
routing`, `qwen36 device-free check: PASS`.

## Consequence that blocks the physical screens

The variant package's execution identity moved from `z31p1j9w…` (the identity
the admitted B=8 receipt records) to `fypi0bf4…`. Two effects:

1. **The admitted B=8 receipt now describes a superseded identity.** Refreshing
   it requires a fresh matched run — which is exactly this change's g5/g6 work,
   and the reason the route stays gated off by default in the meantime.
2. **The receipt builder refuses an admitted input profile.** It requires the
   input profile to be "unadmitted and unmeasured", so once the B=8 binding
   says `physically_admitted = true` there is no way to produce an admission
   receipt for an improvement measurement without temporarily unadmitting the
   binding (or introducing a separate measurement profile).

That second point is a decision about admission integrity, not a coding detail,
so the physical screens are held here rather than worked around.

## Still open

- **g4 physical half**: distinct prompts, unequal lengths, masked tails,
  reordered slots, failed installation against the grouped path on silicon.
- **g5**: simulator coverage or its bounded blocker, then the matched B4
  preservation run on the new identity.
- **g6**: the B8 rescreen under the unchanged gates, pending the profile-state
  decision above.
