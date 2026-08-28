## Context

The prior MTP pilot produced mean draft acceptance `2.4`, median per-prompt speedup `1.202354705288087`, and aggregate speedup `0.9687222124830936`. Its target verifier is exact and gives `1.9469233481156285` width-four end-to-end amortization. The remaining measured draft path calls `Qwen36Model._lm_head()` for every draft. That method all-gathers each vocabulary shard, then `MtpDrafterAdapter._read_token()` transfers one full `vocab_size` row and performs host argmax.

The traced target path already has a narrower mechanism. It returns pre-gather vocabulary shards, computes one local index and value per device, reads only those bounded winners, and resolves cross-shard ties through the pure `select_global_winner()` policy.

## Decision

Add a narrow shared operation to `TracedDecodeAdapter` that selects one greedy token from vocabulary-sharded logits. It will:

1. run the existing local index and maximum reductions on each shard;
2. synchronize before host observation;
3. read one index and one value per physical device;
4. apply the existing deterministic global-winner policy; and
5. release only selector-owned temporaries.

The MTP adapter will support two diagnostic selector routes:

- `full-logit-readback`: the historical all-gather and full-row transfer control;
- `shard-local-top1`: direct sharded LM-head output and bounded winner readback.

The physical probe will warm both routes, then run exact order `full,bounded,bounded,full` for every fixed prompt. It will retain two roots per route and aggregate duplicate roots by median. It will record MTP prefill, draft compute, draft selection, target verification, readback, rollback, cache rebuild, and emitted-token economics separately.

The pure core will reject route drift, missing roots, token or acceptance mismatch, unstable duplicates, invalid timing, invalid rollback, insufficient selector gain, or aggregate speedup not strictly above one. A valid result can select the bounded selector for further MTP work. It cannot admit MTP or traced decode to production.

## Alternatives

### BF4 weight compression

Rejected. Upstream P150x4 evidence measured `+3.8%` to `+7.5%` decode speed, but 256k full-model logits PCC fell to `0.847` or `0.737`. Greedy agreement over 17 tokens did not satisfy the `0.99` logit gate.

### Decode matmul-reduce-scatter fusion

Rejected for this pass. The pinned source contains unused decode fusion helpers, but its own implementation note says the M=1 two-dimensional matmul collapses to about eight cores and loses. The helper was removed from the latest reference tree.

### Runtime uplift

Blocked. The latest pinned source still implements recurrent decode as the same TTNN operation graph. Its fused C++ operation is a chunk-prefill operation, not a single-token decode replacement.

### New fused GDN decode operation

Retained as a later runtime-owned route. It can remove repeated typecast, norm, state-transfer, and recurrence launches, but it requires new TTNN host code, program factories, kernels, bindings, and full numerical validation.

### Confidence-gated or multi-round MTP

Deferred until bounded selection passes. A gate cannot repair a selector whose transfer cost already erases the measured margin. If selection passes, the next discriminating test is a multi-round fixed-prompt run with rejection frequency and sustained emitted-token economics.

## Safety and evidence boundaries

- Keep batch one, three drafts, exact P150x2 devices `0` and `1`, and CPUs `8-31`.
- Preserve target GDN recurrent state, convolution state, paged KV, and MTP KV rollback.
- Keep full target logits available only for diagnostic equivalence checks.
- Count MTP prefill, selection, verification, readback, rollback, and cache rebuild in economics.
- Restore the eager service after every physical attempt.
- Do not claim long-context, batching, TP4, production, or fused-kernel gains from this change.

## Evaluation result

Shard-local selection produced identical tokens, and prompt 2 rollback remained exact. The observed selector reduction was `66.49554434507507%`, with bounded/full ratio `0.3350445565492493`.

The complete candidate remained invalid. Duplicate roots were unstable, bounded prompt 4 had a `0.4443297780235298`-second MTP-prefill outlier, and prompt 2 accepted zero drafts. Aggregate emitted-token speedup was `0.8197972276186697`.

The selector remains a diagnostic building block, not a selected optimization. The next implementation must trace the fixed-shape MTP draft step and replace MTP cache re-prefill with KV snapshot restore. Confidence gating and multi-round economics follow those stability fixes.
