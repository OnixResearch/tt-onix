## Implementation tasks

- [ ] Run the state and identity baseline, then define complete prefix keys and explicit tenant authority. [task:p1] [serial] r[tenstorrent.qwen38_p150x2.hybrid_prefix_identity]
- [ ] Add valid-hit, tokenizer-drift, model-swap, precision-drift, and cross-tenant denial fixtures. [task:p2] [after:p1] r[tenstorrent.qwen38_p150x2.hybrid_prefix_identity]
- [ ] Implement bounded immutable hybrid snapshots and stable-buffer restore with explicit partial-block ownership. [task:p3] [after:p2] r[tenstorrent.qwen38_p150x2.hybrid_prefix_state]
- [ ] Run cold-versus-restored continuation, missing-state, wrong-shape, corrupt-digest, partial-copy, and eviction tests, then rerun the baseline. [task:p4] [after:p3] r[tenstorrent.qwen38_p150x2.hybrid_prefix_state]
- [ ] Measure actual snapshot memory and representative simulator restore, or preserve the exact simulator blocker. [task:p5] [after:p4] r[tenstorrent.qwen38_p150x2.hybrid_prefix_admission]
- [ ] Compare complete cache hit, miss, and eviction costs within an approved physical window and record the bounded verdict. [task:p6] [after:p5] r[tenstorrent.qwen38_p150x2.hybrid_prefix_admission]
