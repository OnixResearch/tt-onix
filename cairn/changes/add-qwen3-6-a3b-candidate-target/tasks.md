# Tasks: Qwen3.6-A3B candidate model target

## Phase 1: Baseline and upstream authority

- [ ] [serial] B1 Record baseline model-target checks, profile freshness, report self-test, and Cairn validation before edits. r[tenstorrent.model_targets.qwen3_6_candidate_report]
- [ ] [serial] A1 Pin the Knight-Ops authority manifest with URL, branch `qwen3_6_a3b-bringup`, commit `37743f58d443be1eaf70ed10eb37b5377dc56b70`, subtree path, and BLAKE3 content identity. r[tenstorrent.model_targets.qwen3_6_upstream_authority]
- [ ] [depends:tenstorrent.model_targets.qwen3_6_upstream_authority] A2 Review the HF release and pin `config.json`, `generation_config.json`, tokenizer files, and `model.safetensors.index.json` with revision, license, Nix hashes, and BLAKE3 identities. r[tenstorrent.model_targets.qwen3_6_upstream_authority]
- [ ] [depends:tenstorrent.model_targets.qwen3_6_upstream_authority] A3 Record machine facts: firmware floor 19.5.0, pinned tt-metal python_env, fork base tt-metal identity, and the Qwen3.5/Qwen3.6 naming mismatch. r[tenstorrent.model_targets.qwen3_6_upstream_authority]

## Phase 2: Contract and candidate profile

- [ ] [depends:tenstorrent.model_targets.qwen3_6_upstream_authority] C1 Extend the Nickel contract with the hybrid state kind, MoE layer-mix facts, multi-shard layout, and candidate precision value. Regenerate canonical profile and consumer projection data. r[tenstorrent.model_targets.hybrid_profile_contract]
- [ ] [depends:tenstorrent.model_targets.hybrid_profile_contract] C2 Add the non-runnable `qwen3-6-35b-a3b-p150x1-candidate` profile with metadata artifacts and the single-P150 topology. r[tenstorrent.model_targets.qwen3_6_candidate_profile]
- [ ] [depends:tenstorrent.model_targets.qwen3_6_candidate_profile] C3 Add device-free hybrid adapter facts: equations source, state lifecycle, required-operator inventory with ttsim coverage audit, and candidate tolerances. r[tenstorrent.model_targets.qwen3_6_candidate_profile]

## Phase 3: Candidate report

- [ ] [depends:tenstorrent.model_targets.qwen3_6_candidate_profile] P1 Author the Qwen3.6 facts JSON with all eleven reviewed facts and keep the unreviewed fixture as a negative case. r[tenstorrent.model_targets.qwen3_6_candidate_report]
- [ ] [depends:tenstorrent.model_targets.qwen3_6_candidate_report] P2 Produce the deterministic verdict receipt with the report binary and check it in. r[tenstorrent.model_targets.qwen3_6_candidate_report]

## Phase 4: Verification and documentation

- [ ] [parallel] V1 Add positive fixtures for authority admission, hybrid export, candidate evaluation, and the complete verdict. r[tenstorrent.model_targets.hybrid_profile_contract] r[tenstorrent.model_targets.qwen3_6_candidate_report]
- [ ] [parallel] V2 Add negative fixtures for authority drift, missing state component, runnable Qwen profile, and unreviewed fact. r[tenstorrent.model_targets.qwen3_6_upstream_authority] r[tenstorrent.model_targets.qwen3_6_candidate_report]
- [ ] [serial] D1 Document the reference pin, candidate facts, verdict interpretation, bounded next change, and explicit non-claims. r[tenstorrent.model_targets.qwen3_6_upstream_authority]
- [ ] [depends:tenstorrent.model_targets.qwen3_6_candidate_report] V3 Run formatting, focused package tests, Nix checks, profile and consumer freshness, existing RWKV regressions, Cairn validation, and all lifecycle gates. r[tenstorrent.model_targets.qwen3_6_candidate_report]

## Verification Coverage

- `Scenario: Pinned authority is admitted` -> A1, A2, V1
- `Scenario: Authority drift is rejected` -> A1, V2
- `Scenario: Valid hybrid profile exports canonical data` -> C1, V1
- `Scenario: Hybrid profile with a missing state component fails` -> C1, V2
- `Scenario: Candidate profile evaluates device-free` -> C2, C3, V1
- `Scenario: Runnable Qwen profile is rejected` -> C2, V2
- `Scenario: Candidate artifacts exclude weight bytes` -> A2, C2
- `Scenario: Reviewed facts produce simulator eligibility only` -> P1, P2, V1
- `Scenario: An unreviewed fact produces an exact blocker` -> P1, V2
- `Scenario: Machine facts bind the firmware floor` -> A3, P1
