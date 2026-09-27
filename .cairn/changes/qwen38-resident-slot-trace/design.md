## Context

`begin_slots` released every decode trace before it reset the rows of a new slot group. Within a group, the trace already stays resident across refills: a refill replays the masked prefill traces beside it, and the next decode step restages every input. That coexistence is exact, according to `.cairn/changes/qwen38-continuous-slot-scheduler/evidence/refill-coexistence.md`. It is also no slower. In the s6 staggered roots of 2026-09-27, six requests arrived while others decoded, so they prefilled beside the resident trace, in each of two passes. Eleven of those twelve refills took 0.94–1.03 times their solo prefill time; one took 1.43 times, 0.10 s longer.

An earlier same-width reuse candidate for the group coordinator was rejected on TT-Metal 0.74 (`cairn/archive/2026-08-26-reuse-qwen38-p150x2-same-width-trace`). Its prefill ran eagerly under the parked trace, and TT-Metal's allocation-under-active-trace path cost 1.27 s per B=4 prefill. That exceeded the 0.85 s capture it saved. A masked refill replays traces instead of running eager operators, and the s6 timings above show no such cost.

## Decisions

### Decision: Keep the trace only for masked refills

**Choice:** `plan_slot_group_trace` keeps the resident trace when all of the following hold:

- exactly one trace is resident;
- it has the group's width;
- its binding is stable;
- the masked prefill traces are present and not stale.

A route change releases the trace first through `switch_fusion_mode`. Every row is still reset, and the first decode window restages every input.

**Rationale:** these are the conditions under which a later refill in a group already runs beside the trace. Missing masked traces mean an eager prefill, the 0.74 failure mode, so the trace is released. The decision reuses `plan_same_width_trace_prefill` and its reasons, and adds `eager-prefill-next`.

### Decision: Admit against the deployed package in ABBA order

**Choice:** four roots per workload, each from a fresh process: baseline (the deployed package), candidate, candidate, baseline. The workloads are the prefix-cache screen's greedy w3 and w5, plus a seeded sampled w3. Every gate uses per-turn means of the two roots of each role:

- exact tokens and identical cache use in all four roots;
- at most a quarter of a candidate root's turns capture;
- median wall ratio ≤ 0.95 and worst turn ≤ 1.10;
- median TTFT ratio ≤ 1.05;
- a startup DRAM drop of at most 64 MiB.

**Rationale:** the change must not alter output or cache behavior. It should remove a capture that is a large part of an isolated agent turn, and it must not reintroduce the 0.74 prefill penalty. ABBA means cancel host load that drifts linearly across the roots.

### Decision: Re-admit the prefix cache on the new package

**Choice:** the same screen adds two cold roots of the new package, one before and one after the four roots above. The two candidate roots double as the cached roots of a prefix-cache ABBA group under the unchanged policy v2.

**Rationale:** the prefix-cache receipt admits one exact package. A new package needs its own cache receipt, and sharing the candidate roots costs two roots, not four.

## Risks / Trade-offs

- A trace resident between groups keeps the TT-Metal hazard that a later allocation overlaps its freed intermediates. Structured reads, route changes and eager refills release it first, as they do inside a group.
- The screen measures agent-shaped sequential turns and seeded sampling. Other traffic shapes are not measured.
- The continuous-slot scheduler's own receipt names package `jxhspibs4ml8`. The packages deployed since then were screened only against a cold baseline of their own continuous-slot scheduler, not against the group coordinator.
