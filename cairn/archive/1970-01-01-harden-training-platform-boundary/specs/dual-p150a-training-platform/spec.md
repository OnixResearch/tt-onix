# Dual-P150A Training Platform Specification Delta

## MODIFIED Requirements

### Requirement: The training runtime has one immutable compatibility identity

r[tenstorrent.training_platform.runtime_bundle] The flake MUST provide one reviewed compatibility manifest for every executed runtime component. Production launch MUST match the package-owned manifest and complete offline dependency closure by BLAKE3 identity.

#### Scenario: Reviewed runtime authority is required

r[tenstorrent.training_platform.runtime_bundle.valid]
- GIVEN the package-owned manifest, configuration, image digest, source revisions, driver, firmware, and wheel closure
- WHEN production bundle admission runs
- THEN every identity MUST match the reviewed Nix closure
- AND the launcher MUST use only that exact bundle.

#### Scenario: Unreviewed immutable input is supplied

r[tenstorrent.training_platform.runtime_bundle.invalid]
- GIVEN a caller supplies a different image digest, repository, package name, ABI, source, driver, firmware value, or wheel identity
- WHEN production bundle admission runs
- THEN admission MUST fail before device access
- AND an immutable but unreviewed value MUST NOT satisfy the selected bundle.

#### Scenario: Runtime dependency closure is offline

r[tenstorrent.training_platform.runtime_bundle.offline]
- GIVEN the selected runtime wheel closure is present in the Nix store
- WHEN a platform or LoRA hardware container installs Python packages
- THEN it MUST use only the hash-locked local wheelhouse
- AND it MUST NOT contact a package index or upgrade Pip.

### Requirement: Training configuration is typed and deterministic

r[tenstorrent.training_platform.configuration] The flake MUST author training-platform policy in typed Nickel and export a closed, versioned projection for Rust admission. Rust MUST enforce every root and nested contract before launch.

#### Scenario: Valid training configuration exports deterministically

r[tenstorrent.training_platform.configuration.valid]
- GIVEN a configuration that satisfies every closed Nickel contract
- WHEN Nickel export and Rust admission run twice
- THEN both generated outputs and BLAKE3 identities MUST match
- AND both validators MUST reach the same result.

#### Scenario: Invalid nested configuration fails admission

r[tenstorrent.training_platform.configuration.invalid]
- GIVEN a configuration with a wrong schema, nested unknown field, wrong package identity, wrong ABI, invalid date, unsafe path, unsupported board class, invalid fabric value, incomplete exclusion policy, or inconsistent mesh
- WHEN Nickel export or Rust admission runs
- THEN validation MUST fail before launch
- AND the result MUST name the invalid field.

### Requirement: Dual-P150A preflight fails closed

r[tenstorrent.training_platform.preflight] The production launcher MUST validate unique device identities, count, firmware, driver, ownership, hugepages, descriptor, mesh shape, fabric policy, and health in the same process immediately before device launch.

#### Scenario: Preflight and launch remain one operation

r[tenstorrent.training_platform.preflight.valid]
- GIVEN two unique supported P150A devices are idle and healthy under the admitted host profile
- WHEN the production launcher runs
- THEN it MUST bind a passing preflight result to the launch plan
- AND it MUST start Docker only after that result passes.

#### Scenario: Unsafe or stale hardware state blocks launch

r[tenstorrent.training_platform.preflight.invalid]
- GIVEN a device is missing, duplicated, extra, busy, unhealthy, incompatible, wrongly connected, or unavailable through hugepages
- WHEN the production launcher runs
- THEN it MUST block Docker before device access
- AND it MUST emit a failed preflight result without changing device state.

### Requirement: The launcher isolates the selected runtime

r[tenstorrent.training_platform.launch] The Nix-owned launcher MUST use package-owned authority, same-process preflight, all required devices, named hugepages, the selected descriptor, and the offline wheel closure. It MUST reject ambient and caller-controlled runtime substitutions.

#### Scenario: Launcher preserves runtime isolation

r[tenstorrent.training_platform.launch.isolated]
- GIVEN the package-owned bundle and a passing same-process preflight
- WHEN the operator launches the runtime
- THEN the process MUST use the admitted image and offline dependencies
- AND environment variables, image tags, arbitrary manifests, arbitrary configurations, or unapproved device paths MUST NOT alter the bundle.

#### Scenario: Launch or cleanup fails

r[tenstorrent.training_platform.launch.failure]
- GIVEN preflight, launch, mesh open, execution, mesh close, restoration, or final health cannot complete
- WHEN the launcher exits
- THEN it MUST return failure
- AND a validated terminal receipt MUST record the failed and unobserved stages without false success claims.

### Requirement: Platform evidence is bounded and content-addressed

r[tenstorrent.training_platform.evidence] The platform MUST emit BLAKE3-bound receipts over sanitized portable artifacts. Raw logs, stable device identifiers, credentials, and absolute private paths MUST remain outside the portable bundle.

#### Scenario: Portable evidence excludes private content

r[tenstorrent.training_platform.evidence.valid]
- GIVEN a hardware run has private logs and typed aggregate results
- WHEN evidence generation runs
- THEN the receipt MUST bind only sanitized portable inputs and outputs
- AND referenced artifacts MUST contain no excluded content.

#### Scenario: Every exit has a terminal receipt

r[tenstorrent.training_platform.evidence.terminal]
- GIVEN any stage returns success, failure, interruption, or timeout
- WHEN the launcher exits
- THEN it MUST emit and validate one terminal stage-state receipt
- AND stages that did not run MUST be marked `not_observed`.

#### Scenario: Unsafe artifact content is supplied

r[tenstorrent.training_platform.evidence.invalid]
- GIVEN a receipt or referenced artifact contains excluded content, stale identities, missing stages, invalid hashes, or training claims
- WHEN receipt validation runs
- THEN validation MUST fail with stable diagnostics.

### Requirement: Platform support has complete validation and documentation

r[tenstorrent.training_platform.verification] The change MUST include bypass regressions, nested contract fixtures, offline-install checks, shell checks, NixOS evaluation, sanitized evidence checks, and one bounded dual-P150A smoke.

#### Scenario: Platform validation closes the change

r[tenstorrent.training_platform.verification.complete]
- GIVEN maintainers intend to sync and archive the hardening change
- WHEN the required validation runs
- THEN all device-free checks, package builds, module evaluations, Cairn gates, and the bounded hardware smoke MUST pass
- AND no test MAY contact a package index or publish private evidence.

## ADDED Requirements

### Requirement: The NixOS host profile matches runtime admission

r[tenstorrent.training_platform.host_module] The NixOS module MUST install a driver compatible with the selected training manifest, verify the selected descriptor, configure named hugepages, and expose the documented training example.

#### Scenario: Compatible host profile evaluates

r[tenstorrent.training_platform.host_module.valid]
- GIVEN the selected training manifest and supported kernel package set
- WHEN the training example evaluates
- THEN its driver, descriptor, hugepages, Docker setup, and access policy MUST satisfy runtime admission
- AND the documented flake example attribute MUST exist.

#### Scenario: Incompatible host profile fails evaluation

r[tenstorrent.training_platform.host_module.invalid]
- GIVEN the module selects an incompatible driver, missing descriptor, missing hugepages, or unsupported access policy
- WHEN NixOS evaluation runs
- THEN evaluation MUST fail with a field-specific assertion before activation.
