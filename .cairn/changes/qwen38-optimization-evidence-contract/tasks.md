## Implementation tasks

- [ ] Run the existing receipt-core baseline and define the typed hybrid artifact manifest. [task:e1] [serial] r[tenstorrent.qwen38_p150x2.experiment_identity]
- [ ] Implement content-bound comparison with swapped-head, renamed-path, missing-conversion, and valid-manifest fixtures. [task:e2] [after:e1] r[tenstorrent.qwen38_p150x2.experiment_identity]
- [ ] Define complete timing fields and predeclared cohort limits with positive and incomplete-root rejection tests. [task:e3] [after:e2] r[tenstorrent.qwen38_p150x2.experiment_timing]
- [ ] Implement listener identity, exclusive ownership, timeout quarantine, and explicit outage checks in the shell. [task:e4] [after:e3] r[tenstorrent.qwen38_p150x2.experiment_authority]
- [ ] Run fake-worker success, foreign-listener, abandoned-thread, and recovery-authority rejection tests, then rerun the device-free baseline. [task:e5] [after:e4] r[tenstorrent.qwen38_p150x2.experiment_authority]
- [ ] Record a fresh control cohort only within an authorized window, or preserve the exact authority blocker without a performance claim. [task:e6] [after:e5] r[tenstorrent.qwen38_p150x2.experiment_timing]
