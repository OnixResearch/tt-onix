## Phase 1: Policy and evaluation

- [x] [serial] Add typed rollback profiles, strict parsing, and positive and negative evaluator tests for r[tenstorrent.qwen38_p150x2.mtp_kv_rollback].
- [x] [serial] Add exact route, state, stability, recovery-ratio, and non-admission gates for r[tenstorrent.qwen38_p150x2.mtp_kv_rollback].

## Phase 2: Device-resident rollback

- [x] [serial] Snapshot the bounded MTP K/V rows and prefix feedback before traced drafting for r[tenstorrent.qwen38_p150x2.mtp_kv_rollback].
- [x] [serial] Restore rejected rows, advance the committed target token, and compare complete next-round state with re-prefill for r[tenstorrent.qwen38_p150x2.mtp_kv_rollback].
- [x] [serial] Add an explicit diagnostic command and package integration while production remains eager for r[tenstorrent.qwen38_p150x2.mtp_kv_rollback].

## Phase 3: Validation and evidence

- [x] [serial] Run Ruff, Python compilation, Nickel export, Qwen positive and negative checks, NixOS service checks, and Cairn validation for r[tenstorrent.qwen38_p150x2.mtp_kv_rollback].
- [ ] [serial] During an authorized outage, run the balanced P150x2 control, restore eager service health, and preserve the immutable receipt for r[tenstorrent.qwen38_p150x2.mtp_kv_rollback].
- [ ] [serial] Record either a selected diagnostic rollback mechanism or an exact blocker, sync the requirement, regenerate the artifact manifest, and archive the change for r[tenstorrent.qwen38_p150x2.mtp_kv_rollback].
