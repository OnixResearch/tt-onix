## Phase 1: Layout-selection core

- [ ] [serial] Add the pure projection-layout selection core with bounded shape eligibility for r[tenstorrent.qwen38_p150x2.gdn_decode_projection_layout].
- [ ] [serial] Add positive and negative device-free tests covering eligibility bounds, gate default, and malformed shapes for r[tenstorrent.qwen38_p150x2.gdn_decode_projection_layout].

## Phase 2: Candidate wiring

- [ ] [serial] Wire the env-gated candidate program configs into the GDN TP decode projections through a new adopted-tree patch for r[tenstorrent.qwen38_p150x2.gdn_decode_projection_layout].
- [ ] [serial] Keep prefill and the default path on AUTO and add packaged-check probes for r[tenstorrent.qwen38_p150x2.gdn_decode_projection_layout].

## Phase 3: Physical evidence

- [ ] [serial] Pass Qwen, package, service, Cairn, and repository-quality checks for r[tenstorrent.qwen38_p150x2.gdn_decode_projection_layout].
- [ ] [serial] Run one matched eager-versus-candidate physical pair and record per-component medians for r[tenstorrent.qwen38_p150x2.gdn_decode_projection_layout].
- [ ] [serial] Preserve receipts, sync the requirement, and archive without an admission claim for r[tenstorrent.qwen38_p150x2.gdn_decode_projection_layout].
