# Tasks: Generate model-target consumer identities

## Phase 1: Inventory and schema

- [x] [serial] B1 Record profile freshness, model-target checks, consumer identity locations, and Cairn validation before edits. r[tenstorrent.model_targets.profile_contract]
- [x] [serial] I1 Inventory repeated model, artifact, topology, precision, state, tolerance, and adapter identities across Nix, Rust, Python, C++, fixtures, and documentation. r[tenstorrent.model_targets.single_authority]
- [x] [depends:tenstorrent.model_targets.profile_contract] S1 Extend the Nickel profile contract with the reviewed consumer identity projection and exact lowercase hash grammar. r[tenstorrent.model_targets.profile_contract]
- [x] [depends:tenstorrent.model_targets.profile_contract] S2 Add invalid profile fixtures for uppercase, mixed-case, short, long, and non-hex identities. r[tenstorrent.model_targets.profile_contract]

## Phase 2: Generators and consumers

- [x] [depends:tenstorrent.model_targets.profile_contract] G1 Add pure deterministic renderers for Rust, Python, C++, and Nix data modules. r[tenstorrent.model_targets.consumer_projection]
- [x] [depends:tenstorrent.model_targets.consumer_projection] G2 Add byte-for-byte freshness checks with field-specific drift diagnostics. r[tenstorrent.model_targets.consumer_projection]
- [x] [depends:tenstorrent.model_targets.consumer_projection] C1 Replace repeated runtime, command, harness, LoRA, evidence, and report identity constants with generated data. r[tenstorrent.model_targets.single_authority]
- [x] [depends:tenstorrent.model_targets.single_authority] C2 Bind the canonical profile identity into each built runtime, command, and evidence output. r[tenstorrent.model_targets.single_authority]

## Phase 3: Compatibility and verification

- [x] [parallel] V1 Add positive cross-language fixtures for profile, artifact, topology, state, precision, tolerance, and adapter identity parity. r[tenstorrent.model_targets.consumer_projection]
- [x] [parallel] V2 Add negative fixtures for stale output, missing field, wrong dimension, wrong topology, changed tolerance, wrong profile, and uppercase hash. r[tenstorrent.model_targets.consumer_projection]
- [x] [serial] K1 Verify existing package attributes, commands, model pins, receipt kinds, topology, and numerical boundaries remain unchanged. r[tenstorrent.model_targets.single_authority]
- [x] [serial] D1 Document the canonical edit, regeneration, review, and freshness workflow for model identity changes. r[tenstorrent.model_targets.profile_contract]
- [x] [depends:tenstorrent.model_targets.consumer_projection] V3 Run formatting, focused model checks, RWKV regressions, Qwen candidate evaluation, Cairn validation, and all change gates. r[tenstorrent.model_targets.device_free_validation]

## Verification Coverage

- `Scenario: Valid profile exports all consumer projections` -> S1, G1, V1
- `Scenario: Invalid identity text fails every boundary` -> S2, V2
- `Scenario: Generated consumer projection is fresh` -> G2, V1
- `Scenario: Generated consumer projection drifts` -> G2, V2
- `Scenario: Runtime and classifier share generated authority` -> C1, C2, V1
- `Scenario: Consumer identity mismatch fails construction` -> C2, V2
- `Scenario: Existing RWKV interfaces remain stable` -> K1, V3
