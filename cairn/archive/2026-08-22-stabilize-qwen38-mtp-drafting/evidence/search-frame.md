## Goal and acceptance boundary

The goal is to reduce and stabilize fixed-shape MTP draft decode without changing target outputs, MTP weights, greedy selection, or production behavior. A local replay gain is insufficient. Selection requires stable matched P150x2 roots, exact bindings and outputs, exact target rollback, and aggregate emitted-token speedup above one.

## Source evidence

- Package-matched `models/demos/blackhole/qwen36/tt/attention/tp.py` owns internal `k_caches` and `v_caches`.
- `TPAttention.forward_decode` receives an explicit position tensor and updates that cache position before SDPA.
- The current MTP drafter runs two identical-shape decode steps after one 128-token prefill bucket.
- The bounded selector already avoids full-vocabulary all-gather and reads one index and value per shard.
- The target width-four verifier is exact and amortizes target execution by `1.9469233481156285x`.

## Mechanism portfolio

| Family | Mechanism | Evidence | Disposition |
|---|---|---|---|
| MTP trace | Capture embedding, fusion, one MTP layer, normalization, local LM head, and bounded reduction | Fixed shapes and explicit device inputs exist in package source | selected for physical control |
| Full cache copy | Clone the maximum-length MTP K/V tensors before each round | Cache tensors span the configured sequence capacity | rejected as avoidable traffic |
| Bounded capture check | Compare only the written K/V row after warmup and capture | Same token, feedback, and position should overwrite the same row exactly | implemented; physical proof pending |
| Logical rollback | Reuse accepted rows and overwrite the first rejected row at its explicit position | Package source supports the position mechanism | later hypothesis; not claimed here |
| Confidence width | Choose zero, one, or three drafts from MTP confidence | Prior prompt 2 accepted zero drafts | deferred until trace timing is stable |
| Fused GDN | Add a new recurrent GDN runtime operation | Larger runtime project and not specific to MTP variance | retained as a later runtime candidate |

## Simulator disposition

The pinned `ttsim` documentation does not establish host-side TTNN trace capture and replay as supported evidence. A simulator run cannot establish silicon trace-region capacity or timing. No simulator result is used for this change.

## Terminal state

Device-free checks and the matched physical control completed. The trace preserved outputs and reduced median draft time, but all ten prompt-route groups failed the `1.25` duplicate-root gate. Traced drafting is not selected, MTP remains unadmitted, and production remains eager.
