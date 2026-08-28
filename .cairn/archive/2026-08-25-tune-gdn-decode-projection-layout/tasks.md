## Phase 1: Evidence-backed target selection

- [x] [serial] Reproduce the decode-component attribution twice on P150x2 and confirm the GDN mixer dominates (582 us x48, 51 percent of a 54.23 ms step) for r[tenstorrent.qwen38_p150x2.gdn_decode_projection_layout].
- [x] [serial] Split the GDN mixer interior (qkvz projection, ab projection, conv, recurrence, out projection, all-reduce) with phase timers inside one bounded window so the candidate targets the dominating op for r[tenstorrent.qwen38_p150x2.gdn_decode_projection_layout].

## Phase 2: Gated candidate

- [x] [serial] Add the pure shape-eligibility policy for the selected op's decode layout candidate behind a default-off environment gate for r[tenstorrent.qwen38_p150x2.gdn_decode_projection_layout].
- [x] [serial] Wire the candidate through an adopted-tree patch with positive and negative packaged-check probes while prefill and defaults stay untouched for r[tenstorrent.qwen38_p150x2.gdn_decode_projection_layout].
- [x] [serial] Pass Qwen, package, service, Cairn, and repository-quality checks for r[tenstorrent.qwen38_p150x2.gdn_decode_projection_layout].

## Phase 3: Physical pair

- [x] [serial] Run matched eager-versus-candidate component probes in one authorized window and record gdn_mixer median deltas for r[tenstorrent.qwen38_p150x2.gdn_decode_projection_layout].
- [x] [serial] Preserve receipts, sync the requirement, and archive without an admission claim for r[tenstorrent.qwen38_p150x2.gdn_decode_projection_layout].
