# Nanbeige4.3-33B on P150x2: the batch-8 defect

Batch 8 is a size the packaged `tt_transformers` accepts (`supported_batches` is
`{1, 2, 4, 8, 16, 32}`) but that produces wrong tokens on this runner path after row
4. The runner refuses it. This page is the single record of what is measured, what is
excluded, what the cause is, and what applying a fix would cost. The primary evidence
is in `.cairn/changes/bring-up-nanbeige43-33b-p150x2/`; task ids below refer to its
`tasks.md`, and its `evidence/bring-up-2026-10-25.md` carries the narratives.

## Symptom

Decode at batch 8 splits the rows: rows 0-4 reproduce the admitted stream exactly,
rows 5-7 diverge from the first decoded token. Prefill is exact for all eight rows, so
every row starts from the same first token. Decode speed is unaffected (14.2 tokens/s
per user, 114.0 aggregate). Batch 1, 2, 4, 16 and 32 are all correct.

## Reproducing the readout

`bench --cache-probe {prefill,decode}` reads layer 0's paged K and V back after
prefill, or after one decode step, and records per row the page-table block, the slot
and the first two values at that slot for every kv head of both tensors. Reaching batch
8 needs the refusal in `ACCEPTED_BATCHES` widened for the run; the shipped command
refuses it on purpose, and the device-free check enforces that refusal (P46).

The reading that localises the fault (P49):

| row | K, four heads | V, four heads |
|---|---|---|
| 0-4 | correct | correct |
| 5 | `[0,0]` four times | correct |
| 6 | `[-6.375,0.125]`, `[-6.25,0]`, `[6.375,0]`, `[-4.5,0]` | correct |
| 7 | four values of order 1e16 | correct |

V is correct on every row and every head, so the update operation's block addressing,
core mapping, batch indexing and per-head loop all work at batch 8. Only the K payload
is stale, always on all four heads.

## What is excluded, and by what

| Claim | Test | Result |
|---|---|---|
| The SDPA core allocation causes it | two per-head core caps, both regimes | bit-identical failure (P13) |
| The row-to-core mapping is wrong | all rows forced onto row 0's page table | all rows identical (P14) |
| A row reads blocks that were never written | page-table variants, then the cache read-back | refuted: every row's blocks are written (P15, P16, P42) |
| Prefill is involved | read-back after prefill only | prefill is exact for all 8 rows; the padded tail is a deterministic value shared by every row (P48) |
| The decode write never happens for rows 5-7 | read-back before and after the step | refuted: the content changes, so all eight writes run (P48) |
| Positions or the shared position tensor matter | distinct positions per row | same three stale values at new slots (P44) |
| The page table's width or contents matter | 32-, 128- and 256-entry rows, different blocks | same three stale values every time (P45) |
| Tracing matters | `--no-trace` eager decode | same values (P45) |
| The update op's geometry is wrong | dump of the K input's shape, shard grid, page count | 8 shards, 8 pages, `B` = batch: consistent (P41) |
| The fused rotary op mishandles its core rows | wrap the op | the op never fires at decode (P51) |

## Cause

Three measurements finish the job.

1. **K is placed alone on a second core row** (P53). At the fused update call the
   tensors are, at batch 4: Q `(0,0)-(3,0)`, K `(4,0)-(7,0)`, V `(0,0)-(3,0)` — one
   row. At batch 8: Q `(0,0)-(7,0)`, **K `(0,1)-(7,1)`**, V `(0,0)-(7,0)` — K alone one
   row down. All are HEIGHT_SHARDED with shard `[32,128]`. The placement follows
   `to_qk_fused_memory_config`'s `k_start_core = (batch % 8, batch // 8)` with
   `row_size = 8`, so batch 8 is the only admitted size at which K is pushed onto a
   second row while Q and V share the first.

2. **That placement is load-bearing and cannot be moved** (P54). Changing `row_size` to
   4, which would put K on row 2, dies before the update with
   `TT_FATAL: Illegal Runtime Args on 0-2: Number of runtime args cannot be modified`
   (`kernel.cpp:780: user_arg_count == runtime_args.size()`). Runtime args are emplaced
   per shard-grid core while the kernels launch on the cos/sin bounding box, so a core
   outside that box runs without arguments.

3. **The non-fused decode path is correct** (P55). `use_qk_fused` is true for this
   model — it is cleared only for multimodal, HF-style rope, or a prefetcher — and it
   selects both the decode rotary (`_mllama_rope_fused_qk_decode`) and the update (one
   `paged_fused_update_cache` rather than two `paged_update_cache`). Forcing it false
   puts K back on `(0,0)-(7,0)`, makes the rows identical, and produces
   `17782, 152518, 324, 735, 13841, 1429, 9531, 152363` — the first five matching the
   admitted stream exactly — with `passed: true`.

So the defect lives in the qk-fused decode path: its K memory placement combined with
the bounding-box runtime-argument scheme. The non-fused path avoids both.

## What a fix would cost

The non-fused path is correct but slower: **101.77** tok/s aggregate (12.72 per user)
against 114.0 for the broken fused path. Two batch-4 streams give 113.8, and batch 16
gives 225.93. Correcting batch 8 therefore buys no throughput — only the ability to
serve eight users in one stream instead of two batch-4 streams.

The change is one Python line in the packaged model, conditional on the batch size:

```python
# models/tt_transformers/tt/model_config.py
if batch_size == 8:
    self.use_qk_fused = False
```

It is not applied. It would add a batch-conditional code path to a packaged upstream
model for no measured gain, and it would change the identity of the shipped package. If
single-stream eight-user serving is ever required, this is the change to apply, followed
by the usual gates: rows identical, ids equal to the admitted batch-1 stream, and the
accuracy and throughput receipts re-earned at the new identity.

## What is not established

- Whether the fused path could be repaired instead of avoided (the bounding-box launch
  or the runtime-arg emplacement) — that is a C++ change in the packaged stack.
- Whether the same fusion defect affects other models on this stack; it is recorded for
  this model and this path only.
- Whether batch 8 would be correct under a prefetcher. A prefetcher clears
  `use_qk_fused`, which is the non-fused path above, but its own prefill and streaming
  paths were never measured for this model.
