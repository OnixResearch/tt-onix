## Context

The checkpoint stores one MTP layer and no dedicated embedding or LM head. Its tensors are all BF16. `mtp.fc.weight` maps the concatenated embedding and target hidden state from width `10240` to `5120`. The decoder layer is full attention with the same Qwen dimensions as the target.

The packaged Transformers `5.5.4` code ignores `^mtp.*`. It cannot serve as the MTP oracle. The checkpoint config and tensors are primary facts. vLLM commit `7ca49fbe4bab019e55d57cdc4b7fd3d55c67c1a6` and SGLang commit `4cb5aebfe08fa0abbd5fbcf84b29ad3d541bd5d3` provide matching inference semantics.

Both references normalize the shifted next-token embedding and target hidden state separately. They concatenate those values, apply `mtp.fc`, run one full-attention Qwen decoder layer, and apply `mtp.norm`. They share the target embedding and LM head. The MTP attention layer owns a KV cache. Prompt or accepted-token target hidden rows must prefill that cache.

## Decision

### Decision: implement a diagnostic first-pass MTP drafter

**Choice:** Load the fifteen `mtp.*` tensors directly from the pinned final safetensor shard. Remap the one decoder layer into the package-matched Qwen full-attention layer. Reuse the target embedding and LM head.

The diagnostic will:

1. Run the target prompt and retain post-final-norm hidden rows.
2. Shift prompt token IDs left and append the target anchor token.
3. Normalize shifted embeddings and target hidden rows separately.
4. Concatenate them and apply the BF16 `mtp.fc` projection.
5. Prefill the MTP decoder layer and its own KV cache.
6. Use the last row to produce draft one.
7. Advance the MTP layer at consecutive positions for drafts two and three.
8. Compare all drafts with the target greedy continuation.
9. Record prompt setup, draft, target verification, acceptance, and projected emitted-token economics separately.

**Rationale:** This is the smallest path that matches the checkpoint and the two independent runtime references. A last-hidden-only shortcut would omit MTP prompt history. A Gemma assistant adapter would use the wrong KV authority.

### Decision: stage rollback after loader and acceptance feasibility

**Choice:** The first bounded physical smoke proves weight loading, cache initialization, draft production, and target alignment. The five-prompt pilot follows only when that smoke passes. Target GDN prefix rollback is required before MTP selection.

**Rationale:** Prefix-state capture changes the validated target trace. It is not justified if the checkpoint drafter cannot fit, align, or meet the measured cost budget.

### Decision: keep MTP weights diagnostic and unadmitted

**Choice:** Production wrappers do not load MTP weights. The explicit probe owns the extra memory and physical outage.

**Rationale:** A valid target base and a runnable drafter are still insufficient without local acceptance, rollback, emitted-token correctness, stable economics, and admission evidence.

## Risks / Trade-offs

- The MTP layer reuses target BF8/BF4 decoder weight layouts after loading BF16 source tensors. This can reduce acceptance.
- A BF16 MTP FC and extra decoder weights increase device memory pressure.
- The target hidden-state contract must be post-final-norm, matching vLLM input preparation.
- Shifted token and hidden rows can be off by one while still producing plausible drafts.
- MTP KV rollback needs accepted-token re-prefill. Target paged KV can be overwritten, but target GDN state needs explicit prefix restoration.
- Host draft-token readback is conservative and can consume the measured budget.

## Safety and evidence boundaries

The probe is text-only, batch one, physical P150x2, and diagnostic-only. It uses three drafts and at least five fixed prompts after a successful smoke. It records exact package, runtime, source, checkpoint, tokenizer, MTP tensor shapes, physical devices, CPU affinity, and service restoration. External acceptance and GPU runtime results are context only.

## Evaluation result

The physical pilot accepted 12 of 15 drafts. It restored one rejected prefix exactly with trace-time GDN snapshots and MTP cache re-prefill.

The candidate was rejected. Its aggregate emitted-token speedup was `0.9687222124830936`, and its draft spread was `2.059840539341421`. Production remains eager.
