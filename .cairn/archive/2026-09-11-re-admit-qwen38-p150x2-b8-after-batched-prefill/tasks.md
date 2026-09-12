# Tasks: Re-admit B=8 under batched prefill

## Phase 1: Preconditions

- [x] [serial] A1 Confirm the sibling change `lift-gdn-grid-instance-cap` passed its physical parity screen and record its receipt identity. r[tenstorrent.qwen38_p150x2.b8_batched_prefill_readmission]
- [x] [serial] A2 Re-read the rejected B=8 receipt and restate every bound, the corpus, and the root order without editing any value. r[tenstorrent.qwen38_p150x2.b8_batched_prefill_readmission]
- [x] [serial] A3 Confirm the profile export is deterministic and byte-identical to the pre-change export. r[tenstorrent.qwen38_p150x2.b8_batched_prefill_readmission.invalid]

## Phase 2: Preparation

- [x] [serial] I1 Rebuild the package with the batched-prefill mechanism and record the new execution identity. r[tenstorrent.qwen38_p150x2.b8_batched_prefill_readmission]
- [x] [serial] I2 Add negative contract tests that reject an edited threshold, a shortened corpus, a reduced root count, and a changed order. r[tenstorrent.qwen38_p150x2.b8_batched_prefill_readmission.invalid]
- [x] [serial] I3 Confirm the B=4 profile, contracts, binding, and receipt are byte-identical to their pre-change state. r[tenstorrent.qwen38_p150x2.b8_batched_prefill_readmission]

## Phase 3: Verification

- [x] [parallel] V1 Run the package build, the profile export check, and the receipt self-tests at both widths. r[tenstorrent.qwen38_p150x2.b8_batched_prefill_readmission.valid]
- [x] [serial] V2 Run a fresh Blackhole simulator smoke before hardware. r[tenstorrent.qwen38_p150x2.b8_batched_prefill_readmission.valid]
- [x] [serial] V3 Run the matched physical roots in exact `single,batch,batch,single` order on devices 0 and 1 with the eight-prompt corpus. r[tenstorrent.qwen38_p150x2.b8_batched_prefill_readmission.valid]
- [x] [serial] V4 Verify every correctness gate: token parity, text parity, distinct state slices, and clean reset. r[tenstorrent.qwen38_p150x2.b8_batched_prefill_readmission.valid]
- [x] [serial] V5 Evaluate every performance bound and record the verdict without editing a bound. r[tenstorrent.qwen38_p150x2.b8_batched_prefill_readmission.rejection]
- [x] [serial] V6 Restore the production service and confirm readiness and health. r[tenstorrent.qwen38_p150x2.b8_batched_prefill_readmission]

## Phase 4: Completion

- [x] [serial] D1 Move the B=8 binding only on a fully passing receipt; otherwise preserve an exact rejected receipt. r[tenstorrent.qwen38_p150x2.b8_batched_prefill_readmission.rejection]
- [x] [serial] D2 Record the comparison with the prior rejection and every non-claim. r[tenstorrent.qwen38_p150x2.b8_batched_prefill_readmission]
- [x] [serial] D3 Run the repository Cairn, artifact, and Qwen package gates. r[tenstorrent.qwen38_p150x2.b8_batched_prefill_readmission]

## Verification Coverage

- `Scenario: Every bound passes and B=8 is admitted` -> V1, V2, V3, V4, V5, D1, V6
- `Scenario: Any bound fails and the rejection stands` -> V5, D1, D2
- `Scenario: A weaker or edited bound is rejected before device work` -> A3, I2, V1
