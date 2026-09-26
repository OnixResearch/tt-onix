## Phase 1: Implementation

- [x] [serial] I1 Widen the masked prefill logit tile to 32 rows, read rows 1–31 for structured reads, and add `prefill_masked_read` with its eager fallback. r[tenstorrent.qwen38_p150x2.structured_reads]
- [x] [serial] I2 Add the batched read traces (4 sequences, 128/256 buckets) and capture them after every body compiles. r[tenstorrent.qwen38_p150x2.structured_reads]
- [x] [serial] I3 Add `/v1/systemone`, the pure read policy, and worker read jobs that merge into batched replays. r[tenstorrent.qwen38_p150x2.structured_reads]
- [x] [serial] I4 Fill the first draw with a placeholder and refine it, sample later draws, memoize prefills, and gate the auto policy past the first slot. r[tenstorrent.qwen38_p150x2.structured_read_draws]
- [x] [serial] I5 Read through the last page-table row, save the carried GDN state after a full first chunk into buffers allocated before capture, restore it for a read with the same first chunk, and drop it before any prefill writes its blocks. r[tenstorrent.qwen38_p150x2.structured_read_prefix]

## Phase 2: Verification

- [x] [parallel] V1 Add device-free probes for slots, read rows, memoization, refinement, sampled draws, stages, batching, worker merging, the selector, and refusals. r[tenstorrent.qwen38_p150x2.structured_reads.invalid] r[tenstorrent.qwen38_p150x2.structured_read_draws.valid]
- [x] [serial] V2 Compare read rows with independent prefills, batched reads with single reads, and generation with the previous package on P150x2. r[tenstorrent.qwen38_p150x2.structured_reads.valid]  (evidence/structured-reads.md)
- [x] [serial] V3 Measure accuracy, latency, and throughput per policy on the packaged service, and exercise the extensions and error paths. r[tenstorrent.qwen38_p150x2.structured_read_draws.valid]  (evidence/structured-reads.md)
- [x] [serial] V4 Compare reused-prefix reads with uncached reads after generation, batched reads, a short read, and a full-width prefill, and time both on P150x2. r[tenstorrent.qwen38_p150x2.structured_read_prefix.valid] r[tenstorrent.qwen38_p150x2.structured_read_prefix.invalid]  (evidence/structured-reads.md)

## Verification Coverage

- `Scenario: A decision reads its slots from one prefill` -> V2
- `Scenario: An unreadable request is refused` -> V1, V3
- `Scenario: Draws that cannot differ cost one read` -> V1
- `Scenario: A reused prefix reads the same logits` -> V4
- `Scenario: An overwritten prefix is not reused` -> V1, V4
