## Phase 1: Acceptance-rule core

- [x] [serial] Add the pure device-free acceptance-rule core (exact, lenient, relaxed, greedy) with per-position realized total-variation reporting for r[tenstorrent.qwen38_p150x2.mtp_exact_rejection_sampling].
- [x] [serial] Add positive and negative device-free tests for the acceptance core, including boundary and malformed-input cases, for r[tenstorrent.qwen38_p150x2.mtp_exact_rejection_sampling].

## Phase 2: Rule admission gating

- [x] [serial] Route sampled acceptance through closed diagnostic-only rule admission while greedy exact-match stays the only production-eligible rule for r[tenstorrent.qwen38_p150x2.mtp_exact_rejection_sampling].

## Phase 3: Evidence

- [x] [serial] Pass Qwen, package, service, Cairn, and repository-quality checks for r[tenstorrent.qwen38_p150x2.mtp_exact_rejection_sampling].
- [x] [serial] Sync the requirement, regenerate the manifest, and archive the change for r[tenstorrent.qwen38_p150x2.mtp_exact_rejection_sampling].
