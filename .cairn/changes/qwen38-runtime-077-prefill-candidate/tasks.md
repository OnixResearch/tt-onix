## Implementation tasks

- [ ] Run the current package baseline and inspect the complete dependency diff for the pinned 0.77 candidate. [task:r1] [serial] r[tenstorrent.qwen38_runtime.candidate_077_identity]
- [ ] Build an isolated candidate closure with immutable component identities and Nix-generated lock changes only. [task:r2] [after:r1] r[tenstorrent.qwen38_runtime.candidate_077_identity]
- [ ] Add capability tests for fused GDN, masks, state copies, paged KV, fused SwiGLU, and TP2 reductions. [task:r3] [after:r2] r[tenstorrent.qwen38_runtime.candidate_077_prefill]
- [ ] Run positive state-carry and negative signature, dtype, mask-upload, and memory-bound fixtures, then rerun device-free checks. [task:r4] [after:r3] r[tenstorrent.qwen38_runtime.candidate_077_prefill]
- [ ] Run a representative Blackhole simulator path and preserve any exact trace or fabric blocker. [task:r5] [after:r4] r[tenstorrent.qwen38_runtime.candidate_077_admission]
- [ ] Run an authorized matched P150x2 control and candidate screen, then record capability-scoped admission or rejection. [task:r6] [after:r5] r[tenstorrent.qwen38_runtime.candidate_077_admission]
