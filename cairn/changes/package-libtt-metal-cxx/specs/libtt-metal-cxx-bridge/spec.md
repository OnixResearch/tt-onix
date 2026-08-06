# libtt-metal-cxx Bridge Delta

## ADDED Requirements

### Requirement: Approved bridge source authority
r[tenstorrent.rust_ml.libtt_authority] The repository MUST pin `libtt-metal-cxx` by URL, commit, Nix source hash, BLAKE3 tree identity, Cargo lock identity, authorization basis, and license status. Repository-maintainer approval MAY authorize use when the upstream SPDX identifier is unknown. The repository MUST NOT invent an SPDX identifier.

#### Scenario: Approved bridge source is admitted
r[tenstorrent.rust_ml.libtt_authority.valid]
- GIVEN commit `3f62391f9d1cb4d5bdf85266239f63a67f93b016` with matching source and lock identities
- AND repository-maintainer approval for use as open-source Tenstorrent work
- WHEN source admission runs
- THEN it produces one deterministic admitted authority
- AND it records the unknown SPDX identifier as metadata debt

#### Scenario: Bridge source or lock drifts
r[tenstorrent.rust_ml.libtt_authority.drift]
- GIVEN a source tree or Cargo lock identity that differs from the pin
- WHEN source admission runs
- THEN admission fails and names the mismatched identity
- AND no package output is exposed

### Requirement: Nix-owned bridge package
r[tenstorrent.rust_ml.libtt_package] The `libtt-metal-cxx` package MUST use Nix-owned Rust, C++, Cargo dependencies, and package-matched TT-Metal paths. It SHALL use the pinned upstream Cargo lockfile. It MUST reject ambient install paths and incompatible TT-Metal headers, libraries, runtime data, or symbols.

#### Scenario: Package builds against packaged TT-Metal
r[tenstorrent.rust_ml.libtt_package.valid]
- GIVEN the admitted bridge source and package-matched TT-Metal outputs
- WHEN Nix builds and links the package
- THEN every required build path resolves inside the Nix store
- AND required bridge symbols are present

#### Scenario: Required TT-Metal API is missing
r[tenstorrent.rust_ml.libtt_package.api_drift]
- GIVEN a TT-Metal package without one required header, library, runtime directory, or symbol
- WHEN the compatibility check runs
- THEN it fails with the exact missing API artifact
- AND it does not substitute an ambient path

### Requirement: Bounded safe Rust facade
r[tenstorrent.rust_ml.libtt_facade] The repository MUST expose a bounded Rust facade for the admitted bridge subset. Pure functions SHALL validate dimensions, byte counts, core ranges, operation order, and result sizes before FFI. The imperative shell SHALL own handles, exception conversion, and reverse-order cleanup.

#### Scenario: Valid facade plan executes in order
r[tenstorrent.rust_ml.libtt_facade.valid]
- GIVEN valid device, unit-mesh, buffer, program, kernel, runtime-argument, enqueue, readback, and close inputs
- WHEN the pure core builds an execution plan
- THEN the shell receives one ordered bounded plan
- AND each acquired resource has one cleanup action

#### Scenario: Invalid facade input fails before FFI
r[tenstorrent.rust_ml.libtt_facade.invalid]
- GIVEN an invalid core range, byte count, result size, or closed handle
- WHEN the pure core validates the request
- THEN it returns a typed field-specific error
- AND no FFI operation runs

#### Scenario: Partial acquisition cleans up safely
r[tenstorrent.rust_ml.libtt_facade.partial_cleanup]
- GIVEN a shell failure after some resources are acquired
- WHEN cleanup runs
- THEN it closes each acquired resource once in reverse order
- AND it reports the original error and any cleanup error without panic

### Requirement: Bridge compatibility evidence
r[tenstorrent.rust_ml.libtt_evidence] The package MUST emit one canonical BLAKE3-bound bridge compatibility receipt. The receipt SHALL bind source, authorization, lockfile, toolchains, TT-Metal identity, API subset, checks, execution class, outcome, and exact blockers. Device-free and simulator evidence MUST NOT imply physical execution, placement, or performance.

#### Scenario: Default validation stays device-free
r[tenstorrent.rust_ml.libtt_evidence.device_free]
- GIVEN the default bridge validation gate
- WHEN it builds, links, inspects symbols, and runs pure tests
- THEN it does not open a device or load ttsim
- AND its receipt classifies execution as device-free

#### Scenario: Simulator round trip is classified as simulated
r[tenstorrent.rust_ml.libtt_evidence.simulated]
- GIVEN a passing Blackhole `1x1` ttsim unit-mesh and inline-kernel round trip
- WHEN receipt construction runs
- THEN it binds the simulator and TT-Metal identities
- AND it classifies the result as simulated

#### Scenario: Simulator receipt claims physical execution
r[tenstorrent.rust_ml.libtt_evidence.false_physical]
- GIVEN only device-free and simulator evidence
- WHEN receipt construction requests a physical classification
- THEN construction fails with an evidence-class diagnostic
- AND no physical receipt is emitted

#### Scenario: Compatibility receipt is deterministic
r[tenstorrent.rust_ml.libtt_evidence.deterministic]
- GIVEN identical bridge, toolchain, TT-Metal, and check inputs
- WHEN receipt construction runs twice
- THEN both canonical receipts and BLAKE3 identities match
