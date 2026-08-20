# Tasks: Return Qwen3.8-27B continuous batching as unsupported

## Phase 1: Prerequisite decision

- [x] [serial] A1 Record the physical traced-decode receipt and rejected verdict. r[tenstorrent.qwen38_p150x2.continuous_batching_non_support]
- [x] [serial] A2 Record that the batching design requires an admitted traced execution base. r[tenstorrent.qwen38_p150x2.continuous_batching_non_support]
- [x] [serial] A3 Return the current continuous-batching candidate as unsupported without a throughput claim. r[tenstorrent.qwen38_p150x2.continuous_batching_non_support]

## Phase 2: Support boundary

- [x] [serial] S1 Keep serialized eager request execution unchanged. r[tenstorrent.qwen38_p150x2.continuous_batching_non_support]
- [x] [serial] S2 Add no scheduler, queue, batch-state, trace, or runtime path. r[tenstorrent.qwen38_p150x2.continuous_batching_non_support]
- [x] [serial] S3 Require a new validated Cairn change before future continuous-batching support. r[tenstorrent.qwen38_p150x2.continuous_batching_non_support]

## Phase 3: Evidence and verification

- [x] [parallel] V1 Add the positive unadmitted-base rejection scenario. r[tenstorrent.qwen38_p150x2.continuous_batching_non_support.valid]
- [x] [parallel] V2 Add the negative scenario that a future trace admission does not imply batch support. r[tenstorrent.qwen38_p150x2.continuous_batching_non_support.invalid]
- [x] [serial] D1 Record the prerequisite, production path, and all non-claims. r[tenstorrent.qwen38_p150x2.continuous_batching_non_support]
- [x] [serial] G1 Run Cairn validation, lifecycle gates, sync checks, and the drain evidence guard. r[tenstorrent.qwen38_p150x2.continuous_batching_non_support]

## Verification Coverage

- `Scenario: Traced execution base is unadmitted` -> A1, A2, A3, V1
- `Scenario: A future traced path is admitted` -> S3, V2
