# Dual P150a Training Platform Specification

## Purpose

Defines the `dual-p150a-training-platform` capability.

## Requirements

### Requirement: The training runtime has one immutable compatibility identity

r[tenstorrent.training_platform.runtime_bundle] The flake MUST provide one immutable compatibility manifest for TT-XLA, TT-Forge, TT-MLIR, TT-Metal, PyTorch/XLA, PyTorch, TT-Blacksmith, driver, and firmware identities.

#### Scenario: Immutable compatible bundle is admitted

r[tenstorrent.training_platform.runtime_bundle.valid]
- GIVEN every runtime component has an immutable revision, digest, version, or package hash
- AND the declared ABI, driver, and firmware constraints agree
- WHEN bundle admission runs
- THEN the bundle MUST receive a deterministic BLAKE3 identity
- AND the launcher MUST use only that selected bundle.

#### Scenario: Mutable or incompatible bundle is rejected

r[tenstorrent.training_platform.runtime_bundle.invalid]
- GIVEN any runtime component uses a mutable tag, lacks an immutable identity, or conflicts with another compatibility constraint
- WHEN bundle admission runs
- THEN admission MUST fail before device access
- AND the result MUST identify each conflicting component.

### Requirement: Training configuration is typed and deterministic

r[tenstorrent.training_platform.configuration] The flake MUST author training-platform configuration in typed Nickel and MUST export only the generated data required by the selected upstream runtime.

#### Scenario: Valid training configuration exports deterministically

r[tenstorrent.training_platform.configuration.valid]
- GIVEN a configuration selects the admitted bundle, Blackhole architecture, two devices, `p150_x2`, required hugepages, bounded cache roots, and evidence output
- WHEN configuration export runs twice
- THEN both generated outputs MUST match byte for byte
- AND both outputs MUST bind the same BLAKE3 configuration identity.

#### Scenario: Invalid training configuration fails admission

r[tenstorrent.training_platform.configuration.invalid]
- GIVEN a configuration has an unknown field, mutable identity, invalid path, unsupported device count, wrong architecture, or inconsistent mesh
- WHEN Nickel validation runs
- THEN export MUST fail before launch.

### Requirement: Dual-P150A preflight fails closed

r[tenstorrent.training_platform.preflight] The launcher MUST validate device identity, device count, firmware, driver compatibility, ownership, hugepages, mesh descriptor, mesh shape, fabric policy, and health before a dual-card workload.

#### Scenario: Dual-P150A preflight passes

r[tenstorrent.training_platform.preflight.valid]
- GIVEN two supported Blackhole P150-class devices are idle and healthy
- AND the packaged `p150_x2` descriptor opens a `1x2` mesh with the configured fabric policy
- WHEN the preflight runs
- THEN it MUST return a passing bounded receipt
- AND both devices MUST remain available for the selected launcher.

#### Scenario: Unsafe hardware state blocks launch

r[tenstorrent.training_platform.preflight.invalid]
- GIVEN a selected device is missing, extra, busy, unhealthy, incompatible, wrongly identified, wrongly connected, or unavailable through hugepages
- WHEN the preflight runs
- THEN it MUST block the launch
- AND it MUST not change firmware, topology, ownership, or device state.

### Requirement: The launcher isolates the selected runtime

r[tenstorrent.training_platform.launch] The Nix-owned launcher MUST use the admitted image digest and source revision, pass all required Tenstorrent devices, mount required hugepages, set the selected mesh descriptor, and reject ambient runtime overrides.

#### Scenario: Launcher preserves runtime isolation

r[tenstorrent.training_platform.launch.isolated]
- GIVEN the platform configuration and preflight receipt are current
- WHEN the operator launches the runtime
- THEN the process MUST use the admitted compatibility bundle
- AND ambient `PYTHONPATH`, `TT_METAL_HOME`, image tags, or unapproved device paths MUST NOT alter that bundle.

#### Scenario: Launch or cleanup fails

r[tenstorrent.training_platform.launch.failure]
- GIVEN the runtime cannot start, open the mesh, execute the smoke operation, close the mesh, or restore ownership
- WHEN the launcher exits
- THEN it MUST return failure
- AND the receipt MUST record the failed stage without claiming restoration or health.

### Requirement: Platform evidence is bounded and content-addressed

r[tenstorrent.training_platform.evidence] The platform MUST emit BLAKE3-bound launch-plan and hardware receipts that identify the bundle, configuration, descriptor, selected device roles, commands, stages, results, artifacts, and non-claims.

#### Scenario: Hardware receipt binds bounded evidence

r[tenstorrent.training_platform.evidence.valid]
- GIVEN a dual-card smoke run completes and restores ownership
- WHEN evidence generation runs
- THEN the receipt MUST bind all portable inputs and outputs
- AND it MUST exclude credentials, raw logs, serial numbers, and absolute private paths.

#### Scenario: Stale or overclaiming evidence is supplied

r[tenstorrent.training_platform.evidence.invalid]
- GIVEN a receipt has stale identities, missing stages, invalid artifact hashes, unsafe fields, or training claims
- WHEN receipt validation runs
- THEN validation MUST fail with stable diagnostics.

### Requirement: Platform support has complete validation and documentation

r[tenstorrent.training_platform.verification] The change MUST include positive and negative device-free fixtures, one bounded dual-P150A smoke receipt, focused package validation, NixOS evaluation, and lifecycle gates.

#### Scenario: Platform validation closes the change

r[tenstorrent.training_platform.verification.complete]
- GIVEN maintainers intend to sync and archive the platform change
- WHEN the required validation runs
- THEN all positive fixtures, negative fixtures, package builds, module evaluations, hardware smoke stages, Cairn validation, and lifecycle gates MUST pass.

### Requirement: Platform documentation preserves the claim boundary

r[tenstorrent.training_platform.documentation] The documentation MUST state the exact runtime identity, hardware prerequisites, mesh and cable requirements, launch command, recovery path, evidence location, and platform non-claims.

#### Scenario: Documentation preserves the platform boundary

r[tenstorrent.training_platform.documentation.complete]
- GIVEN an operator follows the dual-P150A training procedure
- WHEN the operator reads the prerequisites and result interpretation
- THEN the documentation MUST distinguish platform readiness from model training correctness, convergence, and performance.
