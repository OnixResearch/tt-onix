## Context

The adopted source at commit `d102e7f4512060e09e580e40cf26a870bfa88c55` declares `FULL_ATTENTION_LAYERS = {3, 7, 11, 15, 19, 23, 27, 31}` in `weight_mapping.py`. No code reads that symbol. The remapper only changes tensor names and layouts.

Runtime layer construction uses `Qwen36ModelArgs.attention_type_list`. The model configuration reads this list from `text_config.layer_types`, and each layer calls `is_full_attention_layer(layer_num)`. The pinned Qwen3.8 configuration has 64 entries, with 48 `linear_attention` values and 16 `full_attention` values. Its recorded BLAKE3 identity is `a19b66576a666e181293906ba585c13edcb2caa946d342a6a42b9d6a873a119d`.

The current fallback is a 32-entry sequence for the older 9B model. It is valid only when the checkpoint omits `layer_types` and declares the same layer count.

## Decisions

### Decision: make layer-type resolution a pure validated function

**Choice:** Add a pure resolver in `model_config.py`. It receives the optional checkpoint list and declared layer count. It returns an immutable complete list or raises a precise configuration error.

**Rationale:** The functional core owns type recognition, count equality, and fallback eligibility. `Qwen36ModelArgs` remains the shell that reads the Hugging Face configuration and supplies explicit values.

### Decision: remove the unused mapper set

**Choice:** Remove `FULL_ATTENTION_LAYERS` from `weight_mapping.py`. Keep the remapper independent of mixer dispatch.

**Rationale:** The symbol has no consumers and describes only the older 32-layer pattern. Its removal cannot change tensor remapping, but it removes a false authority signal.

### Decision: bind tests to the pinned Qwen3.8 pattern

**Choice:** Add direct positive and negative resolver tests to `qwen36-check`. The positive case checks all 64 positions and the 48/16 counts. Negative cases cover missing non-legacy authority, wrong length, and unknown types.

**Rationale:** Direct pure tests prove the dispatch contract without a device or model weights. Existing physical receipts remain evidence for the already executed checkpoint path, not proof for malformed configurations.

### Decision: correct only the reversed historical labels

**Choice:** Correct the archived repeated-sample narrative to match the receipt's named `candidate` and `eager` objects. Do not alter the receipt or its rejection verdict.

**Rationale:** Candidate comparability passed. Eager wall and gap ratios failed. The correction preserves the original measured values and non-selection decision.

## Risks / Trade-offs

- A non-32-layer checkpoint without `layer_types` now fails with a direct error instead of failing later during layer indexing.
- The source patch changes validation only. It does not provide new physical performance or parity evidence.
- The corrected archive changes prose only. The immutable receipt and its BLAKE3 identity remain unchanged.
