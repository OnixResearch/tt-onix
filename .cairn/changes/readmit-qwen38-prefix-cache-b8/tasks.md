# Tasks: Re-admit the hybrid prefix cache in the served B=8 configuration

## Phase 1: Attribution

- [x] [serial] A1 Confirm the deployed configuration pairs the eight-slot continuous-slot profile with the hybrid prefix cache, and that the prefix cache's admission receipt covers neither that profile nor the current package. r[tenstorrent.qwen38_p150x2.hybrid_prefix_admission]
- [x] [serial] A2 Confirm that a prefix-cache root's identity does not bind the scheduler profile the roots served, so a receipt cannot say which scheduler it covered. r[tenstorrent.qwen38_p150x2.hybrid_prefix_admission]

## Phase 2: Implementation

- [x] [serial] I1 Add a schema-5 admission policy that names the continuous-slot profile every root must have served and keeps every schema-2 gate bound. r[tenstorrent.qwen38_p150x2.hybrid_prefix_admission]
- [x] [serial] I2 Make the admission core bind the scheduler profile when the policy requires it, and leave the older schemas validating the committed roots. r[tenstorrent.qwen38_p150x2.hybrid_prefix_admission]
- [x] [serial] I3 Record the served scheduler profile in each root's identity and require the listener to report one. r[tenstorrent.qwen38_p150x2.hybrid_prefix_admission]
- [x] [parallel] I4 Export the schema-5 policy from the package, bind it into the execution identity, and add a device-free probe for the new gate. r[tenstorrent.qwen38_p150x2.hybrid_prefix_admission]

## Phase 3: Verification

- [x] [serial] V1 Run the device-free gates: the Qwen package check and the repository quality gate. r[tenstorrent.qwen38_p150x2.hybrid_prefix_admission]
- [x] [serial] V2 Run the ABBA comparison on the current package with the eight-slot profile at the served limits: four fresh processes per workload, cold, cached, cached, cold. r[tenstorrent.qwen38_p150x2.hybrid_prefix_admission]  (evidence/prefix-cache-abba.md)
- [x] [serial] V3 Re-run the B=8 scheduler admission on the same package, so the served package carries both receipts. r[tenstorrent.qwen38_p150x2.served_feature_receipts]  (evidence/prefix-cache-abba.md)

## Phase 4: Completion

- [x] [serial] D1 Record both receipts, the evidence, and every non-claim. r[tenstorrent.qwen38_p150x2.hybrid_prefix_admission]  (evidence/prefix-cache-abba.md)
- [x] [serial] D2 Move the served instance to the package that carries both receipts. The root-owned drop-in now names a path the operator owns, and the operator's symlink points at the package, so the move needed no root and no password. The served instance reports the admitted identity, the eight-slot profile, and the prefix-cache profile. r[tenstorrent.qwen38_p150x2.served_feature_receipts]  (evidence/prefix-cache-abba.md)

## Phase 5: Re-admission on the promoted package

- [x] [serial] V4 Re-earn the prefix-cache admission on the promoted package `gzwghi7…` and identity `qwnxwc…` with the schema-5 policy, four ABBA roots, and both workloads; receipt `pkgs/qwen36/evidence/2026-10-26/qwen38-prefix-cache-abba-admission-receipt.json`, BLAKE3 `11288cec…`; all eight gates pass. The 2026-10-21 receipt is superseded as current and retained as history. r[tenstorrent.qwen38_p150x2.hybrid_prefix_admission]  (evidence/re-admission-promoted-2026-10-26.md)
