# Tasks: Return Qwen3.8-27B P150x4 as unsupported

## Phase 1: Availability decision

- [x] [serial] A1 Record the available Tenstorrent device nodes. r[tenstorrent.qwen38_p150x4.availability_non_support]
- [x] [serial] A2 Record the operator decision that four devices are unavailable. r[tenstorrent.qwen38_p150x4.availability_non_support]
- [x] [serial] A3 Record an exact, time-bound unsupported verdict without a performance claim. r[tenstorrent.qwen38_p150x4.availability_non_support]

## Phase 2: Support boundary

- [x] [serial] S1 Keep the admitted P150x2 deployment unchanged. r[tenstorrent.qwen38_p150x4.availability_non_support]
- [x] [serial] S2 Add no P150x4 package, service, profile, or runtime path. r[tenstorrent.qwen38_p150x4.availability_non_support]
- [x] [serial] S3 Require a new validated Cairn change before future P150x4 support. r[tenstorrent.qwen38_p150x4.availability_non_support]

## Phase 3: Evidence and verification

- [x] [parallel] V1 Add the positive absent-topology rejection scenario. r[tenstorrent.qwen38_p150x4.availability_non_support.valid]
- [x] [parallel] V2 Add the negative scenario that future hardware alone does not imply support. r[tenstorrent.qwen38_p150x4.availability_non_support.invalid]
- [x] [serial] D1 Record hardware facts, the operator decision, and all non-claims. r[tenstorrent.qwen38_p150x4.availability_non_support]
- [x] [serial] G1 Run Cairn validation, lifecycle gates, and sync checks. r[tenstorrent.qwen38_p150x4.availability_non_support]

## Verification Coverage

- `Scenario: Required physical topology is absent` -> A1, A2, A3, V1
- `Scenario: Future hardware becomes available` -> S3, V2
