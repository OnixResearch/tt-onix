## Context

The full traced candidate replays embedding, 64 hybrid decoder layers, final norm, vocabulary-sharded LM head, and bounded shard reduction. Each decoder layer contains an attention norm, one mixer, residual addition, feed-forward norm, a dense MLP, and a second residual addition. The corrected checkpoint has 48 GDN mixers, 16 full-attention mixers, and 64 MLPs.

Static tensor accounting estimates that MLP weights represent `57.6%` of decode traffic, GDN weights `27.9%`, full-attention weights `8.2%`, and the LM head `6.2%`. Those values are hypotheses. They do not measure runtime overlap, CCL cost, state updates, kernel choice, or effective bandwidth.

## Decisions

### Decision: capture each coherent stage around real model components

**Choice:** Load the complete pinned model once. Use real layers `0`, `32`, and `60` for GDN, layers `3`, `31`, and `63` for full attention, and layers `0`, `31`, and `63` for MLP. Capture each stage around the same persistent hidden input and prompt-bound state.

**Rationale:** This method preserves package-selected weights, memory layouts, norms, collectives, recurrent state, and paged KV behavior. Three positions reduce the risk that one DRAM placement controls the result.

### Decision: include norms and residual work with their owner stage

**Choice:** A mixer trace includes its input norm, mixer, and residual addition. An MLP trace includes its feed-forward norm, MLP, and residual addition. The LM-head trace includes final norm, the vocabulary-sharded matmul, and bounded shard reduction.

**Rationale:** These are the executable component boundaries in `Qwen36DecoderLayer`. Splitting below them would change layouts and add artificial materialization.

### Decision: measure host costs outside device replay

**Choice:** Measure synchronized mutable-input update and bounded winner readback separately. Synchronize the update before the device timer. Run one unmeasured replay and readback before retained samples. Measure complete replay after the update is complete.

**Rationale:** This prevents queued input copies from being charged to the trace. The result keeps host control costs separate from device execution.

### Decision: restore recurrent state outside retained timing

**Choice:** Snapshot prompt-bound GDN state once. Restore it before each GDN or full-model replay, then synchronize before timing starts. Release each trace before later allocations.

**Rationale:** Every retained replay starts from the same state and uses stable device addresses. Restore and capture setup remain visible metadata but do not enter component medians.

### Decision: require additive closure before selection

**Choice:** The pure core takes the median across samples, then the median across representative layers. It scales GDN, MLP, and full-attention medians by the checkpoint counts. It adds embedding and LM-head medians, then compares that model with matched full replay. Selection requires stable groups, configured closure, a minimum share, and a dominance ratio over the second component.

**Rationale:** Independent component traces can differ from the complete trace because cache residency and scheduling can interact. A model that does not close cannot support causal component shares.

## Risks / Trade-offs

- Component traces do not prove that a proposed kernel change improves full-model decode.
- The additive model can fail even when each direct component timing is stable.
- Restoring all GDN states increases run time but stays outside retained replay timing.
- One bounded run cannot establish thermal, concurrency, batch, or long-context behavior.
- The production service must stop for physical access and must return to eager health after cleanup.
