# Tasks: Enable Qwen3.8 BF4 weight cache families

## Phase 1: Policy and attribution

- [ ] [serial] A1 Confirm the three existing gates, their cache names, and the default BFP8 bins they must never overwrite. r[tenstorrent.qwen38_p150x2.bf4_weight_cache_candidates]
- [ ] [serial] A2 Record the per-family byte reduction and the predicted step-time change against the 28.8 ms weight-bound floor. r[tenstorrent.qwen38_p150x2.bf4_weight_cache_candidates]
- [ ] [serial] A3 Set the parity corpus, the accuracy corpus, and the accuracy floor in the Nickel profile before any screen runs. r[tenstorrent.qwen38_p150x2.bf4_weight_cache_candidates.valid]
- [ ] [serial] A4 Record the published BF16-versus-FP8 MTP acceptance gap as the stated reason token parity alone is insufficient. r[tenstorrent.qwen38_p150x2.bf4_weight_cache_candidates]

## Phase 2: Implementation

- [ ] [serial] I1 Add the pure cache-name and gate policy core with closed validation. r[tenstorrent.qwen38_p150x2.bf4_weight_cache_candidates.cache]
- [ ] [serial] I2 Generate each `.bfp4` cache outside every measured window and never substitute it inside one. r[tenstorrent.qwen38_p150x2.bf4_weight_cache_candidates]
- [ ] [serial] I3 Keep BFP8 selected for every missing, unknown, or malformed gate value. r[tenstorrent.qwen38_p150x2.bf4_weight_cache_candidates]
- [ ] [serial] I4 Bind the gate policy, the profile, and the cache identities into the package execution identity. r[tenstorrent.qwen38_p150x2.bf4_weight_cache_candidates]

## Phase 3: Verification

- [ ] [parallel] V1 Add positive device-free probes for every valid family, cache name, and gate value. r[tenstorrent.qwen38_p150x2.bf4_weight_cache_candidates.valid]
- [ ] [parallel] V2 Add negative probes for colliding cache names, a missing suffix, unknown gate values, and a changed default bin. r[tenstorrent.qwen38_p150x2.bf4_weight_cache_candidates.cache]
- [ ] [serial] V3 Prove every default BFP8 cache is byte-identical after each BF4 run. r[tenstorrent.qwen38_p150x2.bf4_weight_cache_candidates.cache]
- [ ] [serial] V4 Run the package build and the device-free check. r[tenstorrent.qwen38_p150x2.bf4_weight_cache_candidates]
- [ ] [serial] V5 Run a fresh Blackhole simulator smoke before hardware. r[tenstorrent.qwen38_p150x2.bf4_weight_cache_candidates]
- [ ] [serial] V6 Run the numerical screen per family: exact greedy token and text parity plus the accuracy floor. r[tenstorrent.qwen38_p150x2.bf4_weight_cache_candidates.valid]
- [ ] [serial] V7 Run the matched physical screen per family and report the step-time ratio beside the predicted byte saving. r[tenstorrent.qwen38_p150x2.bf4_weight_cache_candidates.valid]
- [ ] [serial] V8 Record each family verdict separately and reject any family that fails a gate. r[tenstorrent.qwen38_p150x2.bf4_weight_cache_candidates.invalid]

## Phase 4: Completion

- [ ] [serial] D1 Confirm the accepted `low_precision_non_admission` requirement is untouched and still binds every unadmitted family. r[tenstorrent.qwen38_p150x2.bf4_weight_cache_candidates]
- [ ] [serial] D2 Record every family verdict, the byte-versus-time comparison, and every non-claim; obtain owner review before any production precision change. r[tenstorrent.qwen38_p150x2.bf4_weight_cache_candidates]
- [ ] [serial] D3 Run the repository Cairn, artifact, and Qwen package gates. r[tenstorrent.qwen38_p150x2.bf4_weight_cache_candidates]

## Verification Coverage

- `Scenario: One family passes its numerical and physical screens` -> V1, V3, V4, V5, V6, V7, D2
- `Scenario: A family fails parity, the accuracy floor, or the timing bound` -> V6, V7, V8, D1
- `Scenario: A cache name or gate value is unsafe` -> V1, V2, V3, I1
