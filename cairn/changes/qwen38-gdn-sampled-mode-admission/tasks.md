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
- [x] [serial] S2 Build the device top-K readback shell, then run the distributional screen on P150x2 under authorized qwen38 service control with the runtime mask discipline. Eight roots completed 2026-08-27; receipt rejected on one bound (min per-position PCC 0.8028 vs 0.99). See `evidence/s2-screen-result.md`. r[tenstorrent.qwen38_p150x2.sampled_mode_kernel_admission.valid]
- [x] [serial] S3 Owner review of every threshold constant against the measured screen before any admission decision. Decision recorded 2026-08-27: option 3, the per-position minimum PCC floor is removed as a gate and kept as a reported diagnostic; every other bound stands. See `evidence/s3-decision.md`. r[tenstorrent.qwen38_p150x2.sampled_mode_kernel_admission]
- [x] [serial] S3a Rebuild the package under the revised policy identity and re-run both matched screens for a fresh receipt. Package `hn61wf5g...`, identity `yv113vpa...`; distributional receipt admitted for sampled mode, component gates pass. See `evidence/s3a-screens-and-b4-admission.md`. r[tenstorrent.qwen38_p150x2.sampled_mode_kernel_admission.valid]
- [x] [serial] S3b Run fresh B=4 admission for the changed execution identity with the fused path enabled. Admitted: exact within-path parity, speedup 3.5293, retention 0.8823, TTFT ratio 4.0062, tail 1.1306. See `evidence/s3a-screens-and-b4-admission.md`. r[tenstorrent.qwen38_p150x2.sampled_mode_kernel_admission.valid]
- [ ] [serial] S3 Owner review of every threshold constant against the measured screen before any admission decision. r[tenstorrent.qwen38_p150x2.sampled_mode_kernel_admission]

## Phase 4: Admission and archive

- [x] [serial] D1a Falsify or prove dual-resident decode traces and measure the mode-switch primitive on hardware. Dual residency falsified (second capture corrupts); release-then-recapture exact at 0.39-0.75 s per switch. See `evidence/routing-primitive.md`. r[tenstorrent.qwen38_p150x2.sampled_mode_kernel_admission]
- [x] [serial] D1b Add the pure sampled-mode router core and the adapter `switch_fusion_mode` primitive with device-free probes. r[tenstorrent.qwen38_p150x2.sampled_mode_kernel_admission.valid]
- [x] [serial] D1c Thread request temperature through the server request path, route per request, and switch modes on change. Temperature parses fail-closed in `core.py`; batches route via `resolve_batch_route` (any deterministic request forces the exact-parity path); the adapter switches before capture; `QWEN36_SAMPLED_FUSION` gates sampled routing off by default; evidence carries `serving_route` and `fusion_mode_switched`. r[tenstorrent.qwen38_p150x2.sampled_mode_kernel_admission]
- [ ] [serial] D1d Validate the wired router physically: greedy routing proven live (serving_route off, known off-path tokens, 16.91 tok/s); sampled path is unreachable because the service rejects every temperature != 0 by policy. Blocked on the owner decision in `evidence/d1d-blocker.md`: implement sampling, extend the admission to greedy, or park the enablement. r[tenstorrent.qwen38_p150x2.sampled_mode_kernel_admission]
- [ ] [serial] D1e Deploy the changed identity as a new service generation. r[tenstorrent.qwen38_p150x2.sampled_mode_kernel_admission]
- [ ] [serial] D2 Record the physical result and all non-claims, then archive the change and run lifecycle gates. r[tenstorrent.qwen38_p150x2.sampled_mode_kernel_admission]

## Verification Coverage

- `Scenario: Sampled-mode distributional evidence admits a candidate` -> A1, A2, A3, P1, P2, S1, S2, S3, D1
- `Scenario: Distributional evidence is invalid, short, or below any bound` -> A3, P2, S2, D2
