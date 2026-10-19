# metalium-rs Adoption Delta

## ADDED Requirements

### Requirement: Fork authority and upstream delta
r[tenstorrent.rust_ml.metalium_fork_authority] The repository MUST pin one fork revision of `metalium-rs` and record its upstream base, license, and complete divergence. The fork delta SHALL contain only reviewed changes that are necessary for admission.

#### Scenario: Fork revision is admitted
r[tenstorrent.rust_ml.metalium_fork_authority.valid]
- GIVEN fork commit `8600f145abfc9955fa1238b2f2e9f51d34b3b3d8` based on upstream `b22a245`
- AND the Apache-2.0 license file, the F16 capability test fix, and the `crates/rwkv7` gate
- WHEN source admission runs
- THEN it produces one deterministic fork authority
- AND the recorded delta is exactly those reviewed changes

#### Scenario: Fork or upstream base drifts
r[tenstorrent.rust_ml.metalium_fork_authority.drift]
- GIVEN a fork commit or upstream base that differs from the record
- WHEN source admission runs
- THEN admission fails and names the mismatch
- AND no package output is exposed

### Requirement: Reproducible native package
r[tenstorrent.rust_ml.metalium_package] The `metalium-rs` package MUST use a pinned Rust toolchain with the `riscv32im-unknown-none-elf` standard library and LLVM tools. It MUST build the shipped crates and install at least one runnable artifact. It MUST NOT depend on ambient rustup, a checkout, or network access at build time.

#### Scenario: Package builds and installs the binary
r[tenstorrent.rust_ml.metalium_package.valid]
- GIVEN the pinned fork, toolchain, and MNIST inputs
- WHEN the Nix package builds
- THEN the firmware and shipped crates compile
- AND `bin/tt-mnist` is installed and starts

#### Scenario: Toolchain or input is unpinned
r[tenstorrent.rust_ml.metalium_package.unpinned]
- GIVEN a package that resolves rustc, cargo, the firmware target, or MNIST from an ambient path
- WHEN the package check evaluates
- THEN it fails and names the unpinned input
- AND no package output is exposed

### Requirement: Parity-gated model migration
r[tenstorrent.rust_ml.metalium_migration] A model service MUST NOT move to `metalium-rs` until its native operator coverage, state carry, numerical agreement, topology, and physical evidence pass. Build or simulator success MUST NOT authorize a service switch.

#### Scenario: Model parity is incomplete
r[tenstorrent.rust_ml.metalium_migration.incomplete]
- GIVEN a model with missing native operators, recurrent-state paths, or physical evidence
- WHEN migration admission runs
- THEN it keeps the model on its current runtime
- AND it names each missing gate

#### Scenario: WKV7 recurrence and state carry pass
r[tenstorrent.rust_ml.metalium_migration.recurrence]
- GIVEN the WKV7 host reference and the composed Burn/TT device path
- WHEN a two-token sequence runs on card 0 and the second token consumes the first token's state
- THEN output and state stay within the recorded tolerance of the host reference
- AND the result does not authorize a fused-kernel, layer, or model claim

#### Scenario: Model gates pass
r[tenstorrent.rust_ml.metalium_migration.complete]
- GIVEN native operator coverage, state carry, numerical agreement, topology, and physical evidence for one model
- WHEN migration admission runs
- THEN that model MAY declare a native runtime profile
- AND the evidence remains specific to the tested topology and model revision

### Requirement: Adoption evidence and non-claims
r[tenstorrent.rust_ml.metalium_adoption_evidence] The adoption MUST record the fork identities, validation counts, package output, installed binary, and evidence class. It MUST NOT claim model support, physical execution, performance, or conformance beyond the recorded gates.

#### Scenario: Adoption evidence is recorded
r[tenstorrent.rust_ml.metalium_adoption_evidence.valid]
- GIVEN the fork validation and package build results
- WHEN adoption evidence is written
- THEN it binds the fork commit, source hash, test counts, package output, and binary smoke
- AND it classifies every result as device-free, simulated, or physical

#### Scenario: Build evidence is presented as model support
r[tenstorrent.rust_ml.metalium_adoption_evidence.false_support]
- GIVEN only a package build or simulator result
- WHEN adoption claims a model runtime
- THEN the claim fails with an evidence-class diagnostic
- AND no model support entry is emitted
