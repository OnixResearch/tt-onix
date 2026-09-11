# Tasks: Return the Qwen3.8 runtime uplift as unselected

## Phase 1: Candidate decision

- [x] [serial] A1 Record the current runtime, model source, and fallback paths. r[tenstorrent.qwen38.runtime_uplift_non_admission]
- [x] [serial] A2 Record that no immutable isolated candidate closure was selected. r[tenstorrent.qwen38.runtime_uplift_non_admission]
- [x] [serial] A3 Return this runtime-uplift evaluation as unselected and unadmitted. r[tenstorrent.qwen38.runtime_uplift_non_admission]

## Phase 2: Runtime boundary

- [x] [serial] S1 Keep TT-Metal `0.74.0-rc6` and the current Qwen service unchanged. r[tenstorrent.qwen38.runtime_uplift_non_admission]
- [x] [serial] S2 Keep all compatibility fallbacks and make no lockfile change. r[tenstorrent.qwen38.runtime_uplift_non_admission]
- [x] [serial] S3 Require a new validated Cairn change before a future runtime candidate can be evaluated. r[tenstorrent.qwen38.runtime_uplift_non_admission]

## Phase 3: Evidence and verification

- [x] [parallel] V1 Add the positive incomplete-candidate rejection scenario. r[tenstorrent.qwen38.runtime_uplift_non_admission.valid]
- [x] [parallel] V2 Add the negative scenario that a newer release does not imply admission. r[tenstorrent.qwen38.runtime_uplift_non_admission.invalid]
- [x] [serial] D1 Record current identities, retained fallbacks, and all non-claims. r[tenstorrent.qwen38.runtime_uplift_non_admission]
- [x] [serial] G1 Run Cairn validation, lifecycle gates, sync checks, and the drain evidence guard. r[tenstorrent.qwen38.runtime_uplift_non_admission]

## Verification Coverage

- `Scenario: No complete runtime candidate exists` -> A1, A2, A3, V1
- `Scenario: A newer runtime becomes available` -> S3, V2
