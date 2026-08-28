## Phase 1: Policy and core

- [x] [serial] A1 Add the pure distributional admission core with named bounds and fail-closed validation. r[tenstorrent.qwen38_p150x2.sampled_mode_kernel_admission]
- [x] [serial] A2 Add the typed Nickel policy, contracts, and validated twin owning every bound. r[tenstorrent.qwen38_p150x2.sampled_mode_kernel_admission]
- [x] [serial] A3 Add positive and negative device-free probes to the qwen36 check. r[tenstorrent.qwen38_p150x2.sampled_mode_kernel_admission.valid] r[tenstorrent.qwen38_p150x2.sampled_mode_kernel_admission.invalid]
- [x] [serial] A4 Bind the new policy files into the package execution identity inputs. r[tenstorrent.qwen38_p150x2.sampled_mode_kernel_admission]
- [x] [serial] A5 Mirror the change into both Cairn trees and validate each with its matching CLI and policy. r[tenstorrent.qwen38_p150x2.sampled_mode_kernel_admission]

## Phase 2: Probe harness

- [x] [serial] P1 Add the pure teacher-forced probe builder: full-vocab readbacks to top-K triples, canonical corpus payload, b3sum digest validation, and the deterministic receipt. r[tenstorrent.qwen38_p150x2.sampled_mode_kernel_admission]
- [x] [serial] P2 Record probe receipts for the closed corpus and exact root order, with per-position metrics the pure core consumes. The S3a dist receipt is the recorded probe receipt; see `evidence/s3a-screens-and-b4-admission.md`. r[tenstorrent.qwen38_p150x2.sampled_mode_kernel_admission.valid]

## Phase 3: Physical screen and owner review

- [x] [serial] S1 Rebuild the fused GDN candidate package from `gdn-decode-fusion-candidate.patch` and capture its execution identity. Package `lk1agynj...`, identity `dfdgl49i...`; see `evidence/drain-pass-20260827.md`. r[tenstorrent.qwen38_p150x2.sampled_mode_kernel_admission]
- [x] [serial] S2 Run the distributional screen on P150x2 under authorized qwen38 service control with the runtime mask discipline. Eight roots 2026-08-27; receipt rejected on one bound under the v1 policy. See `evidence/s2-screen-result.md`. r[tenstorrent.qwen38_p150x2.sampled_mode_kernel_admission.valid]
- [x] [serial] S3 Owner review of every threshold constant against the measured screen before any admission decision. Option 3 recorded 2026-08-27; see `evidence/s3-decision.md`. r[tenstorrent.qwen38_p150x2.sampled_mode_kernel_admission]
- [x] [serial] S3a Rebuild the package under the revised policy identity and re-run both matched screens for a fresh receipt. Admitted for sampled mode, component gates pass; see `evidence/s3a-screens-and-b4-admission.md`. r[tenstorrent.qwen38_p150x2.sampled_mode_kernel_admission.valid]
- [x] [serial] S3b Run fresh B=4 admission for the changed execution identity with the fused path enabled. Admitted: exact within-path parity, 3.5293x aggregate, 0.8823 retention; see `evidence/s3a-screens-and-b4-admission.md`. r[tenstorrent.qwen38_p150x2.sampled_mode_kernel_admission.valid]

## Phase 4: Routing and sampled serving

- [x] [serial] D1a Falsify or prove dual-resident decode traces and measure the mode-switch primitive. Dual residency falsified; release-then-recapture exact at 0.39-0.75 s; see `evidence/routing-primitive.md`. r[tenstorrent.qwen38_p150x2.sampled_mode_kernel_admission]
- [x] [serial] D1b Add the pure sampled-mode router core and the adapter `switch_fusion_mode` primitive with device-free probes. r[tenstorrent.qwen38_p150x2.sampled_mode_kernel_admission.valid]
- [x] [serial] D1c Thread request temperature through the server request path, route per request, and switch modes on change. `QWEN36_SAMPLED_FUSION` gates sampled routing off by default. r[tenstorrent.qwen38_p150x2.sampled_mode_kernel_admission]
- [x] [serial] D1f Owner decision option 1: parse gate plus pure sampled-decode core with probes. r[tenstorrent.qwen38_p150x2.sampled_mode_kernel_admission]
- [x] [serial] D1g Sampled trace tail with per-shard `ttnn.topk(k=20)` and per-step seeded sampling in the coordinator. FillPad layout bug found and fixed; see `evidence/d1g-d1h-sampled-serving.md`. r[tenstorrent.qwen38_p150x2.sampled_mode_kernel_admission]
- [x] [serial] D1h Physical sampled-serving validation: greedy parity unchanged, seeded sampled requests reproduce exactly. All checks pass; see `evidence/d1g-d1h-sampled-serving.md`. r[tenstorrent.qwen38_p150x2.sampled_mode_kernel_admission.valid]
- [x] [serial] D1e Deploy the changed identity as a new service generation. System `43sm969d`, identity `llvy8p6f`, greedy smoke exact, gate off; see `evidence/d1e-deploy.md`. r[tenstorrent.qwen38_p150x2.sampled_mode_kernel_admission]
- [x] [serial] D1i Owner act: enable sampled serving. Module option `qwen38.sampledFusion` (tenstorrent.nix `7230603`), enabled on britton-desktop (onix-core `9f2b2dc0`), system `jdni43pz`. Production smoke: sampled route full at 12.88 t/s, seed 42 reproducible, greedy route off with exact tokens at 16.83 t/s. r[tenstorrent.qwen38_p150x2.sampled_mode_kernel_admission]

## Phase 5: Completion

- [x] [serial] D2 Record the physical result and all non-claims, then archive the change and run lifecycle gates. Follow-up work recorded outside this change: retire the legacy `cairn/` tree after a Cairn pin uplift (see `evidence/drain-pass-20260827.md`), and stop local `path:` builds leaking `__pycache__` into execution identities (see `evidence/d1e-deploy.md`). r[tenstorrent.qwen38_p150x2.sampled_mode_kernel_admission]

## Verification Coverage

- `Scenario: Sampled-mode distributional evidence admits a candidate` -> A1, A2, A3, P1, P2, S1, S2, S3, S3a, S3b, D1a, D1b, D1c, D1f, D1g, D1h, D1e, D1i
- `Scenario: Distributional evidence is invalid, short, or below any bound` -> A3, S2, S3a
