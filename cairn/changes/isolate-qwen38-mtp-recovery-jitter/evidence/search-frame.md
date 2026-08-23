# Search frame

## Goal

Find a bounded mechanism that keeps every MTP recovery prompt-route spread at or below `1.25`. Preserve exact next-round state and a snapshot/re-prefill ratio at or below `0.8`.

## Completion evidence

Completion requires one matched physical P150x2 receipt from one loaded process. The receipt must contain six roots per route and prompt, one MTP trace capture, one stable binding, exact state parity, complete cleanup, and passing timing bounds.

## False completion

The following results do not complete the goal:

- post-run outlier removal;
- fewer samples or a weaker bound;
- device-only timing without host-wall timing;
- simulator, source-review, or upstream timing;
- trace capture charged to no route but repeated for each root;
- changed tokens, acceptance, rollback, or production behavior; and
- a median gain with any unstable prompt-route group.

## Audit risks

The main risks are asynchronous queue inheritance, host scheduling, cyclic garbage collection, repeated trace allocation, changing trace bindings, trace-memory pressure, and hidden state leakage.

## Budget

- Sources: the package-matched receipt, package source, package-matched TT-Metal tests, and pinned latest TT-Metal source.
- Retrieval: at most 20 focused file reads or searches.
- Implementation: at most two mechanism rounds.
- Physical work: at most one matched outage control for this change.
- Workers: no subagents. The labeled passes are correlated.
- Terminal outcomes: validated, blocked, exhausted, or user-decision-required.

## Receipt analysis

The version-one receipt contains these failing recovery spreads:

- prompt 1 snapshot: `1.390242`;
- prompt 2 re-prefill: `1.794979`;
- prompt 3 re-prefill: `1.328179`;
- prompt 4 re-prefill: `1.296387`;
- prompt 4 snapshot: `2.198578`; and
- prompt 5 snapshot: `4.030193`.

Prompt 5 snapshot root 7 took `0.072900195` seconds. The same root had an MTP prefill time of `0.071698810` seconds. Snapshot recovery and MTP prefill had Pearson correlation `0.862331` across snapshot roots. This correlation is diagnostic and does not prove one cause.

## Approach registry

| Family | Mechanism | Claim | Evidence | Gap | State | Next check |
|---|---|---|---|---|---|---|
| queue-boundary | Synchronize before and after each timed route | Removes inherited asynchronous work | Package-matched benchmarks synchronize around measured regions | simpler | audit | Keep explicit synchronization in the persistent runner |
| persistent-trace | Reuse one trace and stable buffers across all roots and recovery steps | Removes repeated capture and eager decode allocation | Version one changes trace-buffer identities each root; TTNN supports multiple live trace IDs | simpler | active | Implement one process-owned trace and fail-closed cleanup |
| host-runtime | Limit thread pools or disable cyclic collection | Removes host pauses | The service has many host threads; one root showed two MTP timing spikes | unknown | blocked | Add only if persistent tracing still fails |
| device-profile | Compare device program time with wall time | Attributes host versus device jitter | TT-Metal profiler can record device programs | equivalent | blocked | Use only after a stable wall control or a new attribution request |
| robust-filter | Remove outliers or compare medians only | Makes the old receipt pass | The strict gate forbids unstable duplicate roots | stronger | falsified | None |

## Selected mechanism

The persistent-trace family is the smallest route that changes real recovery execution. Explicit synchronization remains part of its measurement boundary. The host-runtime family remains a second-round option only.

## Adversarial audit

The persistent trace can fail because the target trace and MTP trace coexist. It can also fail if one trace output is overwritten before state capture. The implementation must retain exact bindings, copy feedback before each replay, release partial state, and reject any capture count other than one.

## First physical warmup result

The first bounded outage reached warmup and failed before measured roots. The adapter rejected `MTP trace binding changed before-controlled-replay: kv_buffer_identity`. The eager service was restored and passed its health check.

Source inspection found that `TPAttention.reset_state()` assigns new `k_caches` and `v_caches` tensors. The first persistent trace therefore remained bound to obsolete cache objects.

## Second mechanism round

The first result falsified process-wide trace reuse with allocation-based cache reset. The revised adapter allocates the four MTP cache tensors once. It clones fixed zero sources and copies them into the same cache objects before each prefill. The next check must reject any address drift and preserve exact cache state.
