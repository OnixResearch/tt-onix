# Tasks: Return Qwen3.8-27B low-precision candidates as unadmitted

## Phase 1: Precision decision

- [x] [serial] A1 Record the production service environment and affected BFP8 defaults. r[tenstorrent.qwen38_p150x2.low_precision_non_admission]
- [x] [serial] A2 Record the existing opt-in BF4 gates and missing admission evidence. r[tenstorrent.qwen38_p150x2.low_precision_non_admission]
- [x] [serial] A3 Return every current BF4 candidate as experimental and unadmitted. r[tenstorrent.qwen38_p150x2.low_precision_non_admission]

## Phase 2: Support boundary

- [x] [serial] S1 Keep the production precision and service environment unchanged. r[tenstorrent.qwen38_p150x2.low_precision_non_admission]
- [x] [serial] S2 Add no named BF4 production profile or performance claim. r[tenstorrent.qwen38_p150x2.low_precision_non_admission]
- [x] [serial] S3 Require a new validated Cairn change before future BF4 admission. r[tenstorrent.qwen38_p150x2.low_precision_non_admission]

## Phase 3: Evidence and verification

- [x] [parallel] V1 Add the positive incomplete-evidence rejection scenario. r[tenstorrent.qwen38_p150x2.low_precision_non_admission.valid]
- [x] [parallel] V2 Add the negative scenario that an opt-in flag does not imply admission. r[tenstorrent.qwen38_p150x2.low_precision_non_admission.invalid]
- [x] [serial] D1 Record the production profile, experimental gates, and all non-claims. r[tenstorrent.qwen38_p150x2.low_precision_non_admission]
- [x] [serial] G1 Run Cairn validation, lifecycle gates, sync checks, and the drain evidence guard. r[tenstorrent.qwen38_p150x2.low_precision_non_admission]

## Verification Coverage

- `Scenario: Candidate evidence is incomplete` -> A1, A2, A3, V1
- `Scenario: An experimental BF4 gate exists` -> S3, V2
