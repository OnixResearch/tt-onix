# Nanbeige4.3-33B prefix-cache design

## Status

Design only. Nanbeige has no serving endpoint, so this note records the cache
boundary and the work required before a hardware measurement. No cache code,
runner integration, or device run is claimed.

## Reuse boundary

A prefix entry is identified by the exact token prefix plus:

- checkpoint file digest and revision;
- package/execution identity and TT-Metal/TT-NN runtime identity;
- precision preset (`bfp8` default or `accuracy`/mix);
- mesh and model configuration;
- tenant/session scope.

A different token, checkpoint, preset, or identity is a miss, not a prefix hit.

## Pool and block model

- The existing Nanbeige paged pool is block size 32 (`--page-block 32`); keep
  that granularity for sharing.
- Full 32-token blocks are shared by reference with refcounts. The final partial
  block is copied or recomputed; a decode write must never target a shared
  block.
- Prefix entries own read-only block references. Eviction is LRU over entries
  with no active references; a referenced block cannot be evicted or reused.
- The cache budget must coexist with the batched-prefill pool bound
  (`--page-blocks <= 4096` at batch 32). Reserve prefix blocks explicitly and
  fail closed when the remaining per-row context would exceed `--max-seq-len`.
- The cache is process-local initially; no cross-process persistence or
  cross-tenant sharing is claimed.

## Prefill path

Upstream `models/tt_transformers/tt/generator.py` already accepts
`start_pos`/`num_cached_tokens` for a sequential prefix-cached prefill; the
batched-prefill path is not compatible with cached prefixes. The runner must:

1. resolve the block prefix from the exact-token index;
2. retain the shared page-table blocks and copy/recompute the tail;
3. call the sequential prefill with the full prompt and `start_pos` equal to the
   cached token count;
4. reject trace-enabled prefixes until the trace/cache interaction is proven;
5. fall back to a cold prefill when the upstream path rejects the shape.

## Accounting

Each request records:

- `cached_input_tokens`;
- `new_input_tokens`;
- `output_tokens`;
- prefix hit/miss, entry generation, evicted block count;
- TTFT and decode rate, with the existing contention rule.

Server-side cache evidence is not assumed: the runner owns the cache, so these
counts are local facts.

## Correctness and measurement gates

- Cold vs warm prefix on the same prompt: greedy ids equal, or the first
  divergence point recorded under a predeclared bound.
- Prefill/teacher-forced metrics within the retained Nanbeige gates.
- Batch 1/8/32: cold/warm TTFT and decode measured in one session, paired with
  the batched-prefill control; quote decode rate with every first-token figure.
- Row identity across batch 8/16/32 preserved.
- No claim until the exact package/checkpoint/preset/mesh identity is recorded.

## Open blockers

- No Nanbeige serving endpoint or request lifecycle exists.
- The runner always prefills from token 0; `Session` has no cache index or
  eviction layer.
- A serving change must add the block index, accounting, and the cold/warm
  hardware gates above. Until then the existing `--batch`/`--batched-prefill`
  behavior is unchanged.
