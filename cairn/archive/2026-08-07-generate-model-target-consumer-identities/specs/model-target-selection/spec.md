# Model Target Selection Specification Delta

## MODIFIED Requirements

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

## ADDED Requirements

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
