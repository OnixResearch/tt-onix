## Context

`BatchGenerationCoordinator._take_batch` groups equal `max_tokens`. `_run_group` retains all rows through the same output bound. The source has decode buckets but no mid-group slot refill.

## Decisions

The core models `free`, `prefilling`, `active`, `complete`, and `quarantined` slots. Each slot has a generation ID. Inputs include explicit time, request bounds, route, memory facts, and completion observations. Outputs are decisions and typed effect plans.

One worker executes device effects for every route. HTTP handlers submit work rather than call MTP directly. Queue limits, refill budgets, fairness, and deadlines come from typed Nickel policy.

Start at B4 capacity. Use a stable row-to-slot mapping or an explicit tested permutation. Inactive rows cannot corrupt active state. A new request receives fresh KV, recurrent, convolution, and RNG state.

The first refill candidate must prove prefill coexistence with resident decode buffers. Precompile all eligible shapes and allocate persistent state first. If that proof fails, drain the group and use current recapture behavior.

Do not treat multiple command queues as independent model-state ownership. Transfers can overlap only under explicit event ordering and buffer lifetimes.

## Validation

Pure tests cover staggered arrivals, unequal lengths, cancellations, deadlines, queue overflow, stale completions, and fairness. Fake adapters reject overlapping device ownership and missing reset actions.

The physical workload includes long requests with short arrivals between decode steps. It records queue-inclusive latency, prefill stalls, captures, memory, and delivered-token throughput. Each active user's continuation must match the independent reference.

## Alternatives and rollback

The rejected same-width reuse experiment remains a negative control. Zero captures is not a success criterion without latency benefit. Arbitrary B8/B16 expansion is out of scope.

Keep the current group coordinator behind a construction-time route choice. A scheduler failure quarantines state before fallback. No unverified live trace survives a mode change.
