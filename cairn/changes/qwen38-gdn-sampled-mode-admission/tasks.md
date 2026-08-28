## Phase 1: Policy and core

- [x] [serial] A1 Add the pure distributional admission core with named bounds and fail-closed validation. r[tenstorrent.qwen38_p150x2.sampled_mode_kernel_admission]
- [x] [serial] A2 Add the typed Nickel policy, contracts, and validated twin owning every bound. r[tenstorrent.qwen38_p150x2.sampled_mode_kernel_admission]
- [x] [serial] A3 Add positive and negative device-free probes to the qwen36 check. r[tenstorrent.qwen38_p150x2.sampled_mode_kernel_admission.valid] r[tenstorrent.qwen38_p150x2.sampled_mode_kernel_admission.invalid]
- [x] [serial] A4 Bind the new policy files into the package execution identity inputs. r[tenstorrent.qwen38_p150x2.sampled_mode_kernel_admission]
- [x] [serial] A5 Mirror the change into both Cairn trees and validate each with its matching CLI and policy. r[tenstorrent.qwen38_p150x2.sampled_mode_kernel_admission]
- [ ] [serial] A6 Retire the legacy `cairn/` tree after the flake pin uplift to a native-layout Cairn revision. r[tenstorrent.qwen38_p150x2.sampled_mode_kernel_admission]

## Phase 2: Probe harness

- [x] [serial] P1 Add the pure teacher-forced probe builder: full-vocab readbacks to top-K triples, canonical corpus payload, b3sum digest validation, and the deterministic receipt. r[tenstorrent.qwen38_p150x2.sampled_mode_kernel_admission]
- [ ] [serial] P2 Record probe receipts for the closed corpus and exact root order, with per-position metrics the pure core consumes. r[tenstorrent.qwen38_p150x2.sampled_mode_kernel_admission.valid]

## Phase 3: Physical screen

- [x] [serial] S1 Rebuild the fused GDN candidate package from `gdn-decode-fusion-candidate.patch` and capture its execution identity. Package `lk1agynj...`, identity `dfdgl49i...`; see `evidence/drain-pass-20260827.md`. r[tenstorrent.qwen38_p150x2.sampled_mode_kernel_admission]
- [ ] [serial] S2 Build the device top-K readback shell, then run the distributional screen on P150x2 under authorized qwen38 service control with the runtime mask discipline. r[tenstorrent.qwen38_p150x2.sampled_mode_kernel_admission.valid]
- [ ] [serial] S3 Owner review of every threshold constant against the measured screen before any admission decision. r[tenstorrent.qwen38_p150x2.sampled_mode_kernel_admission]

## Phase 4: Admission and archive

- [ ] [serial] D1 If and only if every bound passes and the owner confirms the thresholds: run fresh B=4 admission for the changed execution identity and wire sampled-only request routing. r[tenstorrent.qwen38_p150x2.sampled_mode_kernel_admission.valid]
- [ ] [serial] D2 Record the physical result and all non-claims, then archive the change and run lifecycle gates. r[tenstorrent.qwen38_p150x2.sampled_mode_kernel_admission]

## Verification Coverage

- `Scenario: Sampled-mode distributional evidence admits a candidate` -> A1, A2, A3, P1, P2, S1, S2, S3, D1
- `Scenario: Distributional evidence is invalid, short, or below any bound` -> A3, P2, S2, D2
