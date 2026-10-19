# metalium-rs Native Stack Delta

## ADDED Requirements

### Requirement: Pinned metalium-rs reference authority
r[tenstorrent.rust_ml.metalium_rs_authority] The repository MUST bind the reviewed `metalium-rs` source to URL, commit, license facts, and upstream pins. The reference authority SHALL cover device, firmware, kernel, and Burn-backend behavior only. It MUST NOT authorize a package, runtime adapter, or model support claim.

#### Scenario: Pinned source is admitted as a reference
r[tenstorrent.rust_ml.metalium_rs_authority.valid]
- GIVEN commit `b22a245db18252b131679aca992dac85c9f7d812`
- AND Apache-2.0 declared in the workspace manifest with no separate license file
- WHEN source admission runs
- THEN it produces one deterministic reference authority
- AND it records the license metadata gap without inventing an SPDX identifier

#### Scenario: Source revision drifts
r[tenstorrent.rust_ml.metalium_rs_authority.drift]
- GIVEN a checkout whose commit differs from the pin
- WHEN source admission runs
- THEN admission fails and names the revision mismatch
- AND no reference or package output is exposed

### Requirement: Upstream validation without local patches
r[tenstorrent.rust_ml.metalium_rs_validation] The evaluation MUST run the pinned upstream validation tier and record the exact commands, counts, and failing targets. It MUST NOT patch the source to obtain a passing result. A failing upstream gate SHALL produce a blocked verdict.

#### Scenario: Validation tier completes
r[tenstorrent.rust_ml.metalium_rs_validation.valid]
- GIVEN the pinned specification, ttsim, MNIST, and configuration inputs
- WHEN formatting, clippy, generated-file checks, boundary checks, and workspace tests run
- THEN each result is recorded with its command and outcome
- AND ignored and feature-gated tests are identified

#### Scenario: Upstream test fails
r[tenstorrent.rust_ml.metalium_rs_validation.failure]
- GIVEN `burn-tt --test essentials::dtype_capabilities_advertise_what_the_device_computes_and_no_more` fails at the pinned commit
- WHEN the verdict is constructed
- THEN the failure names the assertion, the contradictory implementation line, and the introducing commit
- AND the source is not patched
- AND the verdict is blocked

### Requirement: Pinned toolchain before packaging
r[tenstorrent.rust_ml.metalium_rs_toolchain] A `metalium-rs` package MUST use a pinned Rust toolchain that carries the `riscv32im-unknown-none-elf` standard library and the LLVM tools required by `tt-firmware-images`. The package MUST NOT depend on an ambient rustup installation or a floating `stable` channel.

#### Scenario: Nix toolchain lacks the firmware target
r[tenstorrent.rust_ml.metalium_rs_toolchain.missing]
- GIVEN the nixpkgs development shell with Rust 1.95.0
- WHEN the firmware build resolves `riscv32im-unknown-none-elf` or `llvm-objcopy`
- THEN the build is blocked with the missing component named
- AND no package is exposed

#### Scenario: Pinned toolchain is available
r[tenstorrent.rust_ml.metalium_rs_toolchain.valid]
- GIVEN a pinned toolchain with the target standard library and LLVM tools
- WHEN the package builds the firmware and shipped crates
- THEN every compiler, target, and tool resolves from the pinned inputs
- AND no ambient rustup path is used

### Requirement: Blocked or eligible evaluation verdict
r[tenstorrent.rust_ml.metalium_rs_verdict] The evaluation MUST produce one verdict of `blocked` or `eligible_for_native_package`. Eligibility requires a hermetic pinned toolchain, a passing upstream validation tier, and no local source patches. The verdict MUST classify every result as device-free, simulated, or physical, and it MUST NOT claim model support, physical execution, placement, or performance.

#### Scenario: Stale upstream test blocks eligibility
r[tenstorrent.rust_ml.metalium_rs_verdict.blocked]
- GIVEN a pinned upstream test failure or an unpinned toolchain
- WHEN verdict construction runs
- THEN the verdict is `blocked`
- AND it names every failed gate and the available evidence

#### Scenario: Passing gates permit a package spike
r[tenstorrent.rust_ml.metalium_rs_verdict.eligible]
- GIVEN a pinned toolchain and a passing upstream validation tier
- WHEN verdict construction runs
- THEN the verdict is `eligible_for_native_package`
- AND it exposes no model runtime or hardware claim

#### Scenario: Simulated evidence is presented as physical
r[tenstorrent.rust_ml.metalium_rs_verdict.false_physical]
- GIVEN only host and simulator evidence
- WHEN verdict construction requests a physical classification
- THEN construction fails with an evidence-class diagnostic
- AND no physical receipt is emitted
