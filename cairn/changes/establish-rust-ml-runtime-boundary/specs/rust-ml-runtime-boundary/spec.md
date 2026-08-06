# Rust ML Runtime Boundary Delta

## ADDED Requirements

### Requirement: Typed runtime capability contract
r[tenstorrent.rust_ml.runtime_contract] The repository MUST define runtime profiles through a typed Nickel contract. Each profile SHALL bind runtime identity, model families, operations, state kinds, dtypes, topologies, execution modes, and adapter revision. The exporter MUST reject contradictory or incomplete capability declarations.

#### Scenario: Valid runtime profile exports canonical data
r[tenstorrent.rust_ml.runtime_contract.valid]
- GIVEN a runtime profile with one identity and a complete capability declaration
- WHEN the Nickel exporter validates the profile
- THEN it writes deterministic canonical data
- AND the Rust core admits only the declared capabilities

#### Scenario: Invalid capability declaration fails
r[tenstorrent.rust_ml.runtime_contract.invalid]
- GIVEN a runtime profile that declares physical execution without a physical evidence class
- WHEN the Nickel exporter validates the profile
- THEN validation fails with a field-specific diagnostic
- AND it does not replace the last valid canonical data

### Requirement: Functional runtime core and imperative adapters
r[tenstorrent.rust_ml.adapter_boundary] Runtime admission, session transitions, and receipt construction MUST remain pure deterministic Rust functions. Backend adapters SHALL own process, environment, filesystem, socket, timeout, and device work. An adapter MUST terminate only a child that it created.

#### Scenario: Complete fake session succeeds
r[tenstorrent.rust_ml.adapter_boundary.complete]
- GIVEN an admitted fake runtime and a valid load, prefill, decode, reset, and close sequence
- WHEN the pure session core evaluates each result
- THEN the session reaches the closed state
- AND the final receipt binds every accepted transition

#### Scenario: Invalid session transition fails
r[tenstorrent.rust_ml.adapter_boundary.invalid_transition]
- GIVEN a closed session
- WHEN a decode or second close request enters the pure session core
- THEN the core rejects the request with a stable transition error
- AND no adapter operation runs

#### Scenario: Current adapters preserve command behavior
r[tenstorrent.rust_ml.adapter_boundary.current_adapters]
- GIVEN the default llama.cpp and persistent P150x2 runtime profiles
- WHEN each adapter constructs its command
- THEN the llama.cpp adapter preserves the current chat defaults
- AND the P150x2 adapter preserves the current topology and dispatch rules

#### Scenario: Adapter cannot terminate an unrelated process
r[tenstorrent.rust_ml.adapter_boundary.process_ownership]
- GIVEN an adapter-owned child and an unrelated process with a similar name
- WHEN the owned child times out
- THEN the adapter terminates and reaps only its owned child
- AND the unrelated process remains unchanged

### Requirement: Fail-closed third-party source admission
r[tenstorrent.rust_ml.source_admission] The repository MUST bind each third-party runtime source to an exact revision, content identity, authorization basis, license status, API scope, and compatibility verdict. An explicit upstream license or repository-maintainer approval MAY authorize use. Missing authorization MUST block source and package outputs. The repository MUST NOT invent an unknown SPDX identifier.

#### Scenario: Licensed source is admitted
r[tenstorrent.rust_ml.source_admission.valid]
- GIVEN a source fixture with an exact revision, matching content identity, explicit license, and reviewed API scope
- WHEN source admission runs
- THEN it produces one deterministic admitted verdict

#### Scenario: Approved source with unknown SPDX is admitted
r[tenstorrent.rust_ml.source_admission.maintainer_approval]
- GIVEN `libtt-metal-cxx` at the reviewed commit
- AND repository-maintainer approval for its use as open-source Tenstorrent work
- WHEN source admission runs
- THEN it produces an admitted verdict under maintainer approval
- AND it records the unknown SPDX identifier as metadata debt

#### Scenario: Source without authority is rejected
r[tenstorrent.rust_ml.source_admission.missing_authority]
- GIVEN a source without an explicit license or repository-maintainer approval
- WHEN source admission runs
- THEN it produces a rejected verdict that names the missing authority
- AND no package or source output is exposed

### Requirement: Common runtime evidence envelope
r[tenstorrent.rust_ml.runtime_evidence] Each runtime attempt MUST emit a canonical BLAKE3-bound receipt. The receipt SHALL bind runtime profile, adapter revision, model identity, operations, state kind, topology, execution classification, result, and exact blocker. A receipt MUST NOT infer physical execution from device-free or simulator evidence.

#### Scenario: Device-free receipt is deterministic
r[tenstorrent.rust_ml.runtime_evidence.device_free]
- GIVEN the same fake runtime profile, request sequence, and adapter results
- WHEN receipt construction runs twice
- THEN both canonical receipts and BLAKE3 identities match
- AND both receipts classify execution as device-free

#### Scenario: False physical classification is rejected
r[tenstorrent.rust_ml.runtime_evidence.false_physical]
- GIVEN a simulator or device-free result without physical evidence
- WHEN receipt construction requests a physical classification
- THEN construction fails with an evidence-class diagnostic
- AND it emits no physical receipt
