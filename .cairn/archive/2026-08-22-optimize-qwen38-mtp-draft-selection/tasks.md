## Phase 1: Policy and causal control

- [x] [serial] P1 Add typed selector policy, paired route order, exact identities, timing groups, and fail-closed thresholds for r[tenstorrent.qwen38_p150x2.mtp_draft_selection].
- [x] [serial] P2 Add a pure evaluator for duplicate-root comparability, route parity, selector gain, rollback, and emitted-token economics for r[tenstorrent.qwen38_p150x2.mtp_draft_selection].

## Phase 2: Shared bounded selector

- [x] [serial] I1 Expose one adapter-owned shard-local greedy selector that reuses deterministic global-winner policy for r[tenstorrent.qwen38_p150x2.mtp_draft_selection].
- [x] [serial] I2 Add full-readback and bounded MTP selector routes with separate compute and selection timing for r[tenstorrent.qwen38_p150x2.mtp_draft_selection].
- [x] [serial] I3 Add the paired physical probe and immutable diagnostic receipt for r[tenstorrent.qwen38_p150x2.mtp_draft_selection].

## Phase 3: Validation and disposition

- [x] [serial] T1 Add positive and negative tests for policy shape, route order, ties, malformed roots, parity, instability, rollback, and economics for r[tenstorrent.qwen38_p150x2.mtp_draft_selection].
- [x] [serial] P3 Run one bounded smoke, then one paired five-prompt P150x2 control if the smoke passes for r[tenstorrent.qwen38_p150x2.mtp_draft_selection].
- [x] [serial] D1 Record the selected route or exact blocker, upstream BF4 rejection, non-claims, and eager restoration for r[tenstorrent.qwen38_p150x2.mtp_draft_selection].
- [x] [serial] V1 Pass formatting, Python, Nickel, package, Nix, Cairn, policy, authority, and artifact gates for r[tenstorrent.qwen38_p150x2.mtp_draft_selection].
