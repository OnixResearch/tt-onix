## Phase 1: Baseline and profile authority

- [x] [serial] Record the current RWKV package, evidence, closure, and device-free results before core changes. r[tenstorrent.model_targets.device_free_validation]
- [x] [serial] Add the typed Nickel model-profile contract and deterministic canonical-data exporter. r[tenstorrent.model_targets.profile_contract]
- [x] [serial] Add positive profiles and negative fixtures for missing pins, unknown state kinds, invalid hashes, and generated-data drift. r[tenstorrent.model_targets.profile_contract]

## Phase 2: Immutable target composition

- [x] [serial] Add the pure Nix `mkModelTarget` constructor for profiles, artifacts, adapters, evidence, and topology. r[tenstorrent.model_targets.target_factory]
- [x] [serial] Expose exact target outputs without a mutable current-model value or arbitrary checkpoint override. r[tenstorrent.model_targets.target_factory]
- [x] [serial] Add positive and negative Nix evaluation tests for compatible targets and every declared mismatch class. r[tenstorrent.model_targets.target_factory]

## Phase 3: Architecture and evidence adapters

- [x] [serial] Define the architecture-adapter contract for equations, layouts, state, operations, references, tolerances, and topology limits. r[tenstorrent.model_targets.architecture_adapter]
- [x] [serial] Add a device-free transformer and KV-cache adapter fixture that does not claim model support. r[tenstorrent.model_targets.architecture_adapter]
- [x] [serial] Extract the pure common evidence envelope and keep filesystem work in thin command shells. r[tenstorrent.model_targets.evidence_contract]
- [x] [serial] Make runtime and evidence packages consume one exported model identity and artifact authority. r[tenstorrent.model_targets.single_authority]
- [x] [serial] Add negative tests for wrong profile identities, family payloads, evidence adapters, and early text publication. r[tenstorrent.model_targets.evidence_contract]

## Phase 4: RWKV compatibility migration

- [x] [serial] Move the existing RWKV checkpoint and tokenizer pins into the first typed model profile. r[tenstorrent.model_targets.rwkv_compatibility]
- [x] [serial] Route the RWKV harness, runtime, command, and classifier through the selected target data. r[tenstorrent.model_targets.rwkv_compatibility]
- [x] [serial] Preserve existing RWKV flake attributes, executable names, model hashes, mesh policy, and closure policy. r[tenstorrent.model_targets.rwkv_compatibility]
- [x] [serial] Run positive compatibility fixtures and negative identity-drift fixtures against retained RWKV evidence. r[tenstorrent.model_targets.rwkv_compatibility]

## Phase 5: Candidate onboarding

- [x] [serial] Add the pure candidate-report core and thin shell for deterministic device-free compatibility reports. r[tenstorrent.model_targets.candidate_report]
- [x] [serial] Cover upstream identity, license, artifacts, tokenizer, state, operators, memory, references, evidence, and topology. r[tenstorrent.model_targets.candidate_report]
- [x] [serial] Evaluate a Qwen3.6 candidate only after an exact reviewed release exists. Otherwise, record `unsupported`. r[tenstorrent.model_targets.no_unverified_candidate_claim]
- [x] [serial] Document how a supported candidate advances to a separate simulator and hardware Cairn. r[tenstorrent.model_targets.no_unverified_candidate_claim]

## Phase 6: Validation and lifecycle

- [x] [serial] Run profile, target-factory, adapter, evidence, candidate-report, and RWKV focused checks without hardware. r[tenstorrent.model_targets.device_free_validation]
- [x] [serial] Run formatting, Rust tests, Clippy, Nix evaluation, closure checks, and the relevant flake checks. r[tenstorrent.model_targets.device_free_validation]
- [x] [serial] Run Cairn validation, gates, and traceability before sync and archive. r[tenstorrent.model_targets.no_unverified_candidate_claim]
