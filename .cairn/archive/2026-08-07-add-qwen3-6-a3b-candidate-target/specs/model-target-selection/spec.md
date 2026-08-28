# Model Target Selection Delta

## ADDED Requirements

### Requirement: Qwen3.6-A3B upstream authority pin
r[tenstorrent.model_targets.qwen3_6_upstream_authority] The repository MUST pin one bring-up source and one Hugging Face release as the Qwen3.6-A3B candidate authority. The pin SHALL bind source URL, branch, commit, subtree path, model ID, revision, license, and content identities.

#### Scenario: Pinned authority is admitted
r[tenstorrent.model_targets.qwen3_6_upstream_authority.valid]
- GIVEN a reference tree that matches the pinned branch, commit, subtree path, and content identity
- AND metadata artifacts that match the pinned HF revision and hashes
- WHEN authority admission runs
- THEN the candidate facts bind one deterministic authority identity

#### Scenario: Authority drift is rejected
r[tenstorrent.model_targets.qwen3_6_upstream_authority.drift]
- GIVEN a reference tree or metadata artifact with an identity that differs from the pin
- WHEN authority admission runs
- THEN admission fails and names the mismatched identity
- AND no candidate fact is published

### Requirement: Hybrid state and MoE profile contract
r[tenstorrent.model_targets.hybrid_profile_contract] The profile contract MUST admit one hybrid state kind that binds KV-cache facts and recurrent-matrix facts in one profile. The contract MUST admit MoE layer-mix facts and MUST reject a hybrid profile that omits either state component.

#### Scenario: Valid hybrid profile exports canonical data
r[tenstorrent.model_targets.hybrid_profile_contract.valid]
- GIVEN a hybrid profile with KV-cache facts, recurrent facts, MoE facts, and a multi-shard layout
- WHEN the profile exporter validates the Nickel source
- THEN it writes deterministic canonical data
- AND existing RWKV and fixture profiles keep their current values

#### Scenario: Hybrid profile with a missing state component fails
r[tenstorrent.model_targets.hybrid_profile_contract.invalid]
- GIVEN a hybrid profile that omits the KV-cache facts or the recurrent facts
- WHEN the profile exporter validates the Nickel source
- THEN validation fails with a field-specific diagnostic
- AND it does not replace the last valid canonical data

### Requirement: Qwen3.6 candidate profile stays non-runnable
r[tenstorrent.model_targets.qwen3_6_candidate_profile] The repository MUST add one Qwen3.6-A3B candidate profile with `runnable = false`. The profile SHALL pin hashed metadata artifacts only and SHALL declare the single-P150 topology and the hybrid adapter facts.

#### Scenario: Candidate profile evaluates device-free
r[tenstorrent.model_targets.qwen3_6_candidate_profile.device_free]
- GIVEN the candidate profile with metadata artifacts and the single-P150 topology
- WHEN the target factory evaluates the profile
- THEN evaluation exposes profile identity and candidate facts
- AND it exposes no runtime, command, or closure output

#### Scenario: Runnable Qwen profile is rejected
r[tenstorrent.model_targets.qwen3_6_candidate_profile.runnable_rejected]
- GIVEN a Qwen hybrid profile with `runnable = true` and unpinned weight shards
- WHEN the target factory evaluates the profile
- THEN evaluation fails before any target output exists

#### Scenario: Candidate artifacts exclude weight bytes
r[tenstorrent.model_targets.qwen3_6_candidate_profile.weights_excluded]
- GIVEN the candidate profile artifacts
- WHEN profile validation runs
- THEN no artifact entry references a weight shard byte payload
- AND the shard index artifact binds the 26 shard file names

### Requirement: Qwen3.6 candidate report verdict
r[tenstorrent.model_targets.qwen3_6_candidate_report] The candidate report MUST produce one deterministic verdict for the reviewed Qwen3.6 facts. The verdict SHALL bind simulator-change eligibility and exact blockers. The verdict MUST NOT expose runnable output or make execution, placement, performance, or support claims.

#### Scenario: Reviewed facts produce simulator eligibility only
r[tenstorrent.model_targets.qwen3_6_candidate_report.complete]
- GIVEN a Qwen3.6 facts file with all eleven facts reviewed
- WHEN the report evaluates the facts
- THEN the verdict marks the candidate eligible for one bounded simulator change
- AND it exposes no runnable output

#### Scenario: An unreviewed fact produces an exact blocker
r[tenstorrent.model_targets.qwen3_6_candidate_report.incomplete]
- GIVEN a Qwen3.6 facts file with one missing or empty fact
- WHEN the report evaluates the facts
- THEN the verdict is unsupported and names the missing fact
- AND it exposes no runnable output

#### Scenario: Machine facts bind the firmware floor
r[tenstorrent.model_targets.qwen3_6_candidate_report.machine_facts]
- GIVEN the reviewed memory and topology facts
- WHEN the report evaluates the facts
- THEN the memory fact records the BFP4 weight budget for one 32 GB P150
- AND the topology fact records the Blackhole firmware floor at 19.5.0
