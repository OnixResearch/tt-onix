# Model Target Selection Specification

## Purpose

Defines the `model-target-selection` capability.

## Requirements

### Requirement: Typed model profiles

r[tenstorrent.model_targets.profile_contract] The repository MUST author each model profile in typed Nickel and export canonical target and consumer identity data. Every hash identity MUST use exact lowercase hexadecimal text.

#### Scenario: Valid profile exports all consumer projections

r[tenstorrent.model_targets.profile_contract.valid]
- GIVEN a complete typed profile with reviewed runtime and evidence identity fields
- WHEN profile export runs
- THEN it MUST produce deterministic canonical target data
- AND it MUST provide the complete input for every generated consumer projection.

#### Scenario: Invalid identity text fails every boundary

r[tenstorrent.model_targets.profile_contract.invalid]
- GIVEN a profile identity that is uppercase, mixed-case, short, long, non-hexadecimal, missing, or inconsistent
- WHEN profile validation runs
- THEN validation MUST fail with a field-specific diagnostic
- AND it MUST not replace the last valid generated data.

#### Scenario: Generated data differs from Nickel source

r[tenstorrent.model_targets.profile_contract.drift]
- GIVEN checked-in target or consumer data that differs from its typed Nickel source
- WHEN freshness validation runs
- THEN validation MUST fail and identify the profile and generated output.

### Requirement: Immutable target factory
r[tenstorrent.model_targets.target_factory] The package set MUST provide a pure Nix constructor that composes one compatible profile, artifact set, architecture adapter, evidence adapter, and topology.

#### Scenario: Compatible target evaluates
r[tenstorrent.model_targets.target_factory.compatible]
- GIVEN compatible profile, artifact, architecture, evidence, and topology inputs
- WHEN `mkModelTarget` evaluates the inputs
- THEN it returns exact runtime, command, evidence, source-boundary, and closure outputs
- AND each output binds the selected target identity

#### Scenario: Incompatible target fails closed
r[tenstorrent.model_targets.target_factory.incompatible]
- GIVEN inputs with an architecture, state, evidence, artifact, precision, or topology mismatch
- WHEN `mkModelTarget` evaluates the inputs
- THEN evaluation fails with a mismatch diagnostic
- AND no runnable target output is exposed

#### Scenario: Mutable checkpoint override is requested
r[tenstorrent.model_targets.target_factory.mutable_override]
- GIVEN an exact model-target output and an environment variable or local path for another checkpoint
- WHEN the target command starts
- THEN it rejects the override before model execution
- AND the selected profile identity remains unchanged

### Requirement: Architecture adapter boundary
r[tenstorrent.model_targets.architecture_adapter] Each runnable target MUST use an architecture adapter that owns model equations, tensor layouts, state lifecycle, required operations, reference logic, and tolerances.

#### Scenario: RWKV adapter declares recurrent state
r[tenstorrent.model_targets.architecture_adapter.rwkv]
- GIVEN the migrated RWKV profile and adapter
- WHEN the target factory validates adapter capabilities
- THEN the adapter declares RWKV equations, recurrent state, tensor rules, and required TT-NN operations

#### Scenario: Transformer fixture declares KV cache
r[tenstorrent.model_targets.architecture_adapter.transformer_fixture]
- GIVEN a device-free synthetic transformer profile and test adapter
- WHEN the target factory validates adapter capabilities
- THEN the adapter declares transformer attention, KV-cache state, tensor rules, and a distinct operation set
- AND the fixture makes no model-support claim

#### Scenario: Architecture adapter is missing
r[tenstorrent.model_targets.architecture_adapter.missing]
- GIVEN a valid profile whose architecture family has no registered adapter
- WHEN the target factory evaluates the profile
- THEN evaluation reports the family as unsupported
- AND no generic forward pass is substituted

### Requirement: Single target authority

r[tenstorrent.model_targets.single_authority] Runtime, command, harness, LoRA, evidence, and report packages MUST consume generated identity data from the same canonical profile. Each built output MUST expose that profile identity.

#### Scenario: Runtime and classifier share generated authority

r[tenstorrent.model_targets.single_authority.shared]
- GIVEN a valid selected target and fresh generated consumer data
- WHEN its runtime, command, and evidence packages build
- THEN every output MUST expose the same profile, model, artifact, topology, state, precision, tolerance, and adapter identities.

#### Scenario: Consumer identity mismatch fails construction

r[tenstorrent.model_targets.single_authority.mismatch]
- GIVEN a consumer with stale generated data or an identity from another profile
- WHEN package construction validates the consumer
- THEN validation MUST fail before a runnable or publishable output is exposed.

### Requirement: Model-aware evidence
r[tenstorrent.model_targets.evidence_contract] The classifier MUST validate a shared safety envelope and the selected architecture payload before it publishes generated text.

#### Scenario: Common and architecture evidence pass
r[tenstorrent.model_targets.evidence_contract.complete]
- GIVEN complete model identity, artifact, topology, process, profiler, reference, and architecture evidence
- WHEN the selected classifier validates the evidence set
- THEN it emits a deterministic passing classification
- AND only then it publishes generated text

#### Scenario: Architecture payload uses the wrong adapter
r[tenstorrent.model_targets.evidence_contract.wrong_adapter]
- GIVEN a valid common envelope and an architecture payload for another adapter
- WHEN the selected classifier validates the evidence set
- THEN validation fails with an adapter-identity diagnostic
- AND it does not publish generated text

#### Scenario: Common evidence is incomplete
r[tenstorrent.model_targets.evidence_contract.incomplete]
- GIVEN evidence with a missing artifact identity, topology fact, profiler attribution, or CPU-fallback fact
- WHEN the selected classifier validates the evidence set
- THEN validation fails without evaluating a success claim
- AND it does not publish generated text

### Requirement: Deterministic candidate report
r[tenstorrent.model_targets.candidate_report] The repository MUST provide a deterministic device-free report for a proposed model profile, adapter set, and topology.

#### Scenario: Candidate facts are complete
r[tenstorrent.model_targets.candidate_report.complete]
- GIVEN reviewed upstream identity, revision, license, artifact, tokenizer, state, operator, memory, reference, evidence, and topology facts
- WHEN the candidate report evaluates the proposed target
- THEN it reports whether the candidate is eligible for a separate bounded simulator change
- AND it exposes no runnable model output

#### Scenario: Candidate facts are incomplete
r[tenstorrent.model_targets.candidate_report.incomplete]
- GIVEN a candidate with a missing pin, license, tokenizer, operator, memory, reference, evidence, or topology fact
- WHEN the candidate report evaluates the proposed target
- THEN it reports `unsupported` or `invalid` with exact blockers
- AND it exposes no runnable model output

### Requirement: RWKV compatibility
r[tenstorrent.model_targets.rwkv_compatibility] The migration MUST preserve the existing RWKV package attributes, executable names, model identities, mesh policy, evidence semantics, and production closure policy.

#### Scenario: Existing RWKV interfaces remain stable
r[tenstorrent.model_targets.rwkv_compatibility.stable]
- GIVEN the migrated RWKV model target
- WHEN existing RWKV package and evidence commands run
- THEN the current flake attributes and executable names resolve to the migrated target
- AND the current model revision, hashes, exact P150x2 mesh, and fail-closed evidence behavior remain unchanged

#### Scenario: RWKV identity drifts during migration
r[tenstorrent.model_targets.rwkv_compatibility.drift]
- GIVEN a migrated RWKV target with a changed model identity, tokenizer identity, mesh policy, or closure policy
- WHEN the compatibility validation runs
- THEN validation fails before the change can replace the existing outputs

### Requirement: Device-free target validation

r[tenstorrent.model_targets.device_free_validation] The repository MUST validate profile export, consumer generation, freshness, target composition, adapters, evidence, candidate reports, and compatibility without acquiring a Tenstorrent device.

#### Scenario: Device-free target suite passes

r[tenstorrent.model_targets.device_free_validation.complete]
- GIVEN canonical profiles and fresh generated consumer projections
- WHEN the focused target suite runs
- THEN all cross-language identity, package, evidence, source-boundary, and compatibility checks MUST run without hardware.

#### Scenario: Target suite tries to open hardware

r[tenstorrent.model_targets.device_free_validation.hardware_access]
- GIVEN the device-free target suite
- WHEN a generation or validation path tries to open a device, stop an owner, or read live machine state
- THEN the suite MUST fail with the forbidden action.

### Requirement: No unverified candidate claim
r[tenstorrent.model_targets.no_unverified_candidate_claim] The repository MUST NOT claim Qwen3.6 availability, compatibility, capacity, execution, placement, or performance without the required upstream and machine evidence.

#### Scenario: Qwen3.6 identity is not reviewed
r[tenstorrent.model_targets.no_unverified_candidate_claim.unreviewed]
- GIVEN a proposed Qwen3.6 profile without an exact reviewed upstream release and complete adapter facts
- WHEN the candidate report evaluates the profile
- THEN it records exact blockers and exposes no runnable target
- AND documentation does not describe the candidate as supported

#### Scenario: Device-free candidate work completes
r[tenstorrent.model_targets.no_unverified_candidate_claim.device_free]
- GIVEN all device-free work in this change is complete
- WHEN the change is ready for archive
- THEN the lifecycle record requires a separate simulator and hardware change for any new model family
- AND it makes no physical execution or performance claim

### Requirement: Consumer projections are deterministic data

r[tenstorrent.model_targets.consumer_projection] The repository MUST generate deterministic Rust, Python, C++, and Nix identity data from the canonical profile. Generated projections MUST contain data only and MUST remain byte-for-byte fresh.

#### Scenario: Generated consumer projection is fresh

r[tenstorrent.model_targets.consumer_projection.valid]
- GIVEN a canonical profile and checked-in consumer projections
- WHEN every projection is regenerated in a temporary directory
- THEN each generated file MUST match its checked-in file byte for byte
- AND each projection MUST bind the same profile BLAKE3 identity.

#### Scenario: Generated consumer projection drifts

r[tenstorrent.model_targets.consumer_projection.invalid]
- GIVEN a checked-in projection with a changed, missing, extra, reordered, or stale identity field
- WHEN freshness validation runs
- THEN validation MUST fail and identify the language output and changed profile.

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
