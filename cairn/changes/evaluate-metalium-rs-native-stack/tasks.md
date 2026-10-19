# Tasks: Evaluate the metalium-rs native stack

## Phase 1: Authority and license facts

- [x] [serial] B1 Record the host, Rust, tt-kmd, ttsim, and flake baseline before the build. r[tenstorrent.rust_ml.metalium_rs_authority]
- [x] [serial] A1 Pin `Knight-Ops/metalium-rs` commit `b22a245db18252b131679aca992dac85c9f7d812` in the reference tree and source registry. r[tenstorrent.rust_ml.metalium_rs_authority]
- [x] [depends:tenstorrent.rust_ml.metalium_rs_authority] A2 Record the workspace Apache-2.0 declaration, the missing license file, and the upstream input pins. r[tenstorrent.rust_ml.metalium_rs_authority]

## Phase 2: Upstream build and validation

- [x] [depends:tenstorrent.rust_ml.metalium_rs_authority] V1 Fetch the pinned specification, ttsim builds, MNIST inputs, and `cfg_defines.h`. r[tenstorrent.rust_ml.metalium_rs_validation]
- [x] [depends:tenstorrent.rust_ml.metalium_rs_validation] V2 Run formatting, clippy with warnings denied, generated-file checks, and dependency-boundary checks. r[tenstorrent.rust_ml.metalium_rs_validation]
- [x] [depends:tenstorrent.rust_ml.metalium_rs_validation] V3 Run `cargo test --workspace --no-fail-fast` and record counts, ignored tests, and failing targets. r[tenstorrent.rust_ml.metalium_rs_validation]
- [x] [depends:tenstorrent.rust_ml.metalium_rs_validation] V4 Reproduce the missing-header failure, fetch `cfg_defines.h`, and rerun its target. Record the remaining upstream failure and its contradictory implementation line. r[tenstorrent.rust_ml.metalium_rs_validation.failure]

## Phase 3: Toolchain and package gate

- [x] [serial] T1 Record the nixpkgs Rust 1.95.0 gap for `riscv32im-unknown-none-elf`, `llvm-objcopy`, and `llvm-objdump`. r[tenstorrent.rust_ml.metalium_rs_toolchain.missing]
- [ ] [serial] T2 Blocked: Add a pinned Nix Rust toolchain with the firmware target and LLVM tools. r[tenstorrent.rust_ml.metalium_rs_toolchain.valid]
- [ ] [serial] T3 Blocked: Add the `metalium-rs` package against the pinned toolchain. r[tenstorrent.rust_ml.metalium_rs_toolchain.valid]
- [ ] [serial] T4 Blocked: Re-evaluate a revision whose upstream validation tier passes. r[tenstorrent.rust_ml.metalium_rs_verdict.eligible]

## Phase 4: Verdict and documentation

- [x] [depends:tenstorrent.rust_ml.metalium_rs_validation] E1 Write the evidence record with environment, commands, counts, blocker, and evidence classes. r[tenstorrent.rust_ml.metalium_rs_verdict.blocked]
- [x] [serial] D1 Document the reference authority, the blocked package verdict, the distinct cxx bridge scope, and the non-claims. r[tenstorrent.rust_ml.metalium_rs_verdict]
- [x] [depends:tenstorrent.rust_ml.metalium_rs_verdict] V5 Run Cairn validation and the lifecycle gates for this change. r[tenstorrent.rust_ml.metalium_rs_verdict]

## Verification Coverage

- `Scenario: Pinned source is admitted as a reference` -> A1, A2
- `Scenario: Source revision drifts` -> A1
- `Scenario: Validation tier completes` -> V1, V2, V3
- `Scenario: Upstream test fails` -> V3, V4, E1
- `Scenario: Nix toolchain lacks the firmware target` -> T1
- `Scenario: Pinned toolchain is available` -> T2, T3
- `Scenario: Stale upstream test blocks eligibility` -> E1, D1
- `Scenario: Passing gates permit a package spike` -> T4
- `Scenario: Simulated evidence is presented as physical` -> E1
