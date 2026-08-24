## Phase 1: Acceptance-rule core

- [ ] [serial] Add the pure device-free acceptance-rule core (exact, lenient, relaxed, greedy) with per-position realized total-variation reporting for r[tenstorrent.qwen38_p150x2.mtp_exact_rejection_sampling].
- [ ] [serial] Add positive and negative device-free tests for the acceptance core, including boundary and malformed-input cases, for r[tenstorrent.qwen38_p150x2.mtp_exact_rejection_sampling].

## Phase 2: Sampled MTP wiring

- [ ] [serial] Wire the exact sampled acceptance into the MTP drafter and selection path behind the existing diagnostic-only gating for r[tenstorrent.qwen38_p150x2.mtp_exact_rejection_sampling].
- [ ] [serial] Keep greedy exact-match acceptance as the production default for r[tenstorrent.qwen38_p150x2.mtp_exact_rejection_sampling].

## Phase 3: Evidence

- [ ] [serial] Pass Qwen, package, service, Cairn, and repository-quality checks for r[tenstorrent.qwen38_p150x2.mtp_exact_rejection_sampling].
- [ ] [serial] Run one authorized bounded P150x2 sampled-MTP case, restore eager health, and preserve the result for r[tenstorrent.qwen38_p150x2.mtp_exact_rejection_sampling].
- [ ] [serial] Sync the requirement, regenerate the manifest, and archive the change for r[tenstorrent.qwen38_p150x2.mtp_exact_rejection_sampling].
