# Tasks: Profile the vLLM TT serving dependencies

## Phase 1: Constraints

- [ ] [serial] A1 Re-read the preflight and restate every version difference, the plugin revision, the pinned vLLM tag, and its unpacked hash. r[tenstorrent.qwen38_p150x2.vllm_serving_dependency_profile]
- [ ] [serial] A2 Record the Starlette security floor and its CVE identifier as a non-relaxable constraint. r[tenstorrent.qwen38_p150x2.vllm_serving_dependency_profile.invalid]
- [ ] [serial] A3 Record the load-bearing serving settings and the misleading failure each one prevents, including disabled prefix caching and the L1 small size. r[tenstorrent.qwen38_p150x2.vllm_serving_dependency_profile.settings]
- [ ] [serial] A4 Record why on-device sampling is refused below four devices, with the logits-per-device figure and the ceiling. r[tenstorrent.qwen38_p150x2.vllm_serving_dependency_profile.settings]

## Phase 2: Implementation

- [ ] [serial] I1 Add a vLLM build with the empty target device against the pinned tag, installing common requirements explicitly. r[tenstorrent.qwen38_p150x2.vllm_serving_dependency_profile]
- [ ] [serial] I2 Add the isolated dependency profile without touching any deployed Qwen dependency. r[tenstorrent.qwen38_p150x2.vllm_serving_dependency_profile.invalid]
- [ ] [serial] I3 Add the pure serving-settings policy core with closed validation and a fail-closed readiness budget. r[tenstorrent.qwen38_p150x2.vllm_serving_dependency_profile.settings]
- [ ] [serial] I4 Install the plugin entry points separately from the vLLM dependency set. r[tenstorrent.qwen38_p150x2.vllm_serving_dependency_profile]

## Phase 3: Verification

- [ ] [parallel] V1 Add positive and negative probes for the settings policy: every required setting present, and each one missing or unsafe. r[tenstorrent.qwen38_p150x2.vllm_serving_dependency_profile.settings]
- [ ] [parallel] V2 Add negative probes for a lowered Starlette floor, a suppressed requirement, and a changed deployed dependency. r[tenstorrent.qwen38_p150x2.vllm_serving_dependency_profile.invalid]
- [ ] [serial] V3 Resolve the profile and record every resolved version beside its target requirement. r[tenstorrent.qwen38_p150x2.vllm_serving_dependency_profile]
- [ ] [serial] V4 Run the ABI coherence check and fail closed on a second Torch or a missing TT-NN extension. r[tenstorrent.qwen38_p150x2.vllm_serving_dependency_profile.invalid]
- [ ] [serial] V5 Run the plugin's own positive and negative tests with its declared pytest constraint. r[tenstorrent.qwen38_p150x2.vllm_serving_dependency_profile]
- [ ] [serial] V6 Run one bounded non-MTP model trial and record startup time, readiness, and the bounded response. r[tenstorrent.qwen38_p150x2.vllm_serving_dependency_profile.valid]
- [ ] [serial] V7 Confirm the deployed Qwen environment and the serialized service are unchanged and healthy. r[tenstorrent.qwen38_p150x2.vllm_serving_dependency_profile.valid]

## Phase 4: Completion

- [ ] [serial] D1 Record the resolved profile, the trial result, and every non-claim, including the absent throughput claim. r[tenstorrent.qwen38_p150x2.vllm_serving_dependency_profile]
- [ ] [serial] D2 If resolution does not close, record the exact unsatisfiable constraint pair instead of forcing an install. r[tenstorrent.qwen38_p150x2.vllm_serving_dependency_profile.invalid]
- [ ] [serial] D3 Run the repository Cairn, artifact, and Qwen package gates. r[tenstorrent.qwen38_p150x2.vllm_serving_dependency_profile]

## Verification Coverage

- `Scenario: The profile resolves and the stack serves one bounded trial` -> V3, V4, V5, V6, V7
- `Scenario: The profile cannot resolve or the ABI is incoherent` -> V2, V4, D2
- `Scenario: A load-bearing serving setting is missing` -> A3, A4, I3, V1
