# g1/g2: derived grouped-prefill eligibility and its fixtures

Recorded 2026-09-11 for change `qwen38-grouped-prefill-b8-rescreen`.

## The defect this replaces

`Qwen36Model.prefill_paged_grouped` carries the group cap as a literal:

```python
gdn_max_bg = 8 if bucket <= 2 * gdn_chunk else 1
group_size = max(1, min(group_size, gdn_max_bg))
```

Its own comment derives that 8 from TP4 geometry — "BH = B*Nv_tp must stay <=
the compute grid (~96-104 cores on P150). With Nv_tp=12 that's B <= 8". At TP2
the local value-head count is 24, not 12, so the same derivation gives 4 on a
one-instance-per-core scan. The literal is therefore wrong for the TP2 service
in the safe direction only by accident: copying it would over-subscribe the
grid (8 x 24 = 192 instances against 110 cores).

Measured context: both physical P150s report 11 x 10 = 110 compute cores, and
`gdn_chunk_size` is 128 ("GDN seq kernel requires 128"), not the fused scan's
internal chunk of 32.

## What was added

**`pkgs/qwen36/service/qwen36_service/grouped_prefill_core.py`** — a pure,
device-free planner. It takes the runtime capability, the local value-head
count, the GDN bucket chunk, the bucket, the scratch budget, and the per-user
slot list, and returns either a grouped plan (with explicit group members and
explicit state destinations) or a per-user decision with the bound that
rejected it. Nothing allocates; every fallback is decided before allocation.

**Runtime capability record** — the package now bakes
`share/qwen36/runtime-capability.json` from what it actually links, using the
same `gdnGridRuntime` declaration that selects the runtime:

| Package | Record |
| --- | --- |
| `qwen38-gdn-grid` | `{"gdn_scan_chunked_instances": true}` |
| `qwen38` (shared) | `{"gdn_scan_chunked_instances": false}` |

So the capability is derived from the linked runtime rather than configured,
and it cannot disagree with the kernels actually shipped. The shared and variant
packages therefore carry different execution identities
(`gycbf6993rv2wck0z499r5rv0pq73wiy` and `wzikjfi7p1d0aym4g349c8c3v5b4kfwp`).

## Derived bounds, and the fixtures that pin them

`head_bound = compute_cores * instances_per_core // local_value_heads`, and
`memory_bound = scratch_budget // bytes_per_user`. `group_size = min(users,
head_bound, memory_bound)`; a result below two users is the per-user route, not
a group.

Device-free fixtures (`probe_grouped_prefill_eligibility`, run inside the
package gate as `PASS grouped prefill eligibility`):

| Fixture | Expectation |
| --- | --- |
| TP2, 4 users, one-instance scan, bucket 128 | grouped, head bound 4, slots `(5,0,7,2)` preserved |
| TP2, 8 users, chunked scan | one group of 8, head bound 9 |
| TP2, 8 users, one-instance scan | **split** into `(0,1,2,3),(4,5,6,7)` — what the literal 8 got wrong |
| TP4, 8 users, one-instance scan | head bound 9, so the literal is wrong at both widths |
| users wider than the whole grid | per-user, `head-capacity-exceeded` |
| scratch budget for one user | per-user, `scratch-budget-exceeded` |
| bucket above 2 x 128 | per-user, `bucket-above-single-pass` |
| bucket exactly 2 x 128 | grouped (boundary held) |
| empty group / single user | per-user, `empty-group` / `single-user` |
| duplicate slots, short length list, negative count, bucket below longest user | rejected before planning |
| capability claiming a chunked scan without an instance width, or zero cores | rejected by capability validation |

## What is not done

- **g3 (wiring)**: the adapter still calls `prefill_paged_slots`. The planner is
  not yet consulted by the service, so no served behavior has changed.
- **g4 (state tests)**: distinct prompts, unequal lengths, masked tails,
  reordered slots, and failed-installation handling against the grouped path.
- **g5/g6 (physical)**: the B4 preservation run and the B8 rescreen. Note that
  B=8 is now admitted on the chunked variant through the sibling re-admission
  change, so g6 is an improvement rescreen under the same unchanged gates
  rather than an admission prerequisite.
