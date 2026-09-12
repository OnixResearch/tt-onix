# Tasks: Make the declared width 16 reachable and measure it

## Phase 1: Attribution

- [x] [serial] A1 Confirm what caps width 16: the closed admission width set, the adapter capacity, and the model's allocated batch size. r[tenstorrent.qwen38_p150x2.b16_execution]
- [x] [serial] A2 Confirm no B=16 profile, corpus, or binary exists and record the B=8 scaffolding to mirror. r[tenstorrent.qwen38_p150x2.b16_execution]

## Phase 2: Implementation

- [x] [serial] I1 Widen the closed width set to include 16 without changing any other bound. r[tenstorrent.qwen38_p150x2.b16_execution.valid]
- [x] [serial] I2 Add the B=16 profile family with pinned contracts and a sixteen-prompt corpus. r[tenstorrent.qwen38_p150x2.b16_execution.invalid]
- [x] [serial] I3 Wire the package: identity inputs, share copies, export, install assertions, and the width-16 evidence binary. r[tenstorrent.qwen38_p150x2.b16_execution.valid]
- [x] [serial] I4 Add device-free probes for the B=16 profile, eligibility, and the rejected mappings. r[tenstorrent.qwen38_p150x2.b16_execution.invalid]

## Phase 3: Verification

- [x] [serial] V1 Run the package build and the device-free check. r[tenstorrent.qwen38_p150x2.b16_execution.valid]
- [x] [serial] V2 Prove on silicon that sixteen concurrent requests execute at width 16. r[tenstorrent.qwen38_p150x2.b16_execution.valid]
- [x] [serial] V3 Run the matched four-root protocol at width 16 and record the receipt or the exact failing gate. r[tenstorrent.qwen38_p150x2.b16_execution.valid]
- [x] [serial] V4 Record the intermittent program error: diagnostic, program id, affected core range, and observed rate. r[tenstorrent.qwen38_p150x2.b16_execution.blocked]
- [x] [serial] V5 Confirm the B=4 and B=8 profiles, bindings, and receipts are unchanged, and that production still serves the single-stream route. r[tenstorrent.qwen38_p150x2.b16_execution]

## Phase 4: Completion

- [x] [serial] D1 Record the result and every non-claim; move the B=16 binding only on a fully passing receipt. r[tenstorrent.qwen38_p150x2.b16_execution]
- [x] [serial] D2 Run the repository Cairn, artifact, and Qwen package gates. r[tenstorrent.qwen38_p150x2.b16_execution]
