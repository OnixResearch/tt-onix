## Context

`Knight-Ops/metalium-rs` at commit `b22a245db18252b131679aca992dac85c9f7d812` is a Rust workspace whose shipped crates are `tt-isa`, `tt-device`, `tt-kmd`, `tt-layout`, `tt-kernels`, `tt-firmware-images`, `burn-tt`, and `tt-mnist`. `tt-ttsim` and `tt-tests` are dev-only. The separate `tt-firmware` workspace targets `riscv32im-unknown-none-elf` and is built by `tt-firmware-images/build.rs`.

The workspace manifest declares Apache-2.0. The pinned tree has no `LICENSE`, `COPYING`, or `NOTICE` file.

Pinned inputs are ttsim `v1.10.9`, tt-kmd `ttkmd-2.11.0`, tt-metal `cfg_defines.h` at `81989dcdb8f9b340c932ae7a71a346f4f08703eb`, and Burn `=0.21.0`. The local host runs tt-kmd `2.10.0`, whose header differs from 2.11.0 only by the additive `SMC_MSG` ioctl and structure. The tree does not call that ioctl, and `PINNED_API_VERSION` is 2, which the loaded driver satisfies.

The repository's Cairn changes do not yet reference this source. `package-libtt-metal-cxx` targets a different API surface: TT-Metal host calls through `cxx`.

## Decisions

### Decision: Establish a reviewed reference authority, not a package authority

**Choice:** Pin one commit and record it as the reviewed native-Rust implementation reference for device, firmware, kernel, and Burn-backend behavior.

The reference authority does not admit a package, a model runtime, or a support claim. Those require the evaluation verdict to be eligible.

**Rationale:** Source review and build evidence are separate facts. Naming the reviewed revision prevents an unpinned branch or a later commit from becoming de facto authority.

### Decision: Evaluate the upstream validation tier without local patches

**Choice:** Fetch the pinned specification, simulator, and MNIST inputs; run formatting, clippy, the generated-file checks, the dependency-boundary checks, and `cargo test --workspace --no-fail-fast` at the pinned revision.

Do not patch the source to make a failing test pass. Record the exact failing target, assertion, and contradictory implementation line.

**Rationale:** An evaluation that silently repairs upstream has not evaluated upstream. A pinned failure is a valid blocked verdict.

### Decision: Keep the native stack distinct from the cxx bridge

**Choice:** Leave `package-libtt-metal-cxx` and `establish-rust-ml-runtime-boundary` in place. This change does not redirect their source pin.

**Rationale:** metalium-rs does not expose TT-Metal host APIs. Replacing the bridge pin with this tree would make the bridge spec unsatisfiable. The two sources serve different contracts.

### Decision: Do not add a Nix package while the toolchain and upstream gates are open

**Choice:** Record the nixpkgs gap. The development-shell Rust 1.95.0 has no `riscv32im-unknown-none-elf` standard library and no `llvm-objcopy` or `llvm-objdump`; the firmware build needs both.

A future package needs a pinned toolchain that carries the target and LLVM tools, and a passing upstream validation tier. The floating `stable` in `rust-toolchain.toml` is also insufficient for package identity.

**Rationale:** A package built with an ambient rustup toolchain is not reproducible. Adding a flake input or a fixed-output toolchain is a separate decision with its own provenance.

### Decision: Classify evidence exactly

**Choice:** The build, clippy, generated-file checks, and simulator-gate tests are device-free or simulated evidence. No physical device was opened. No model inference, placement, performance, or silicon conformance was measured.

**Rationale:** Simulator agreement and host compilation cannot establish hardware behavior.

## Validation Strategy

Positive checks cover the pinned source checkout, formatting, clippy with warnings denied, the four generated-file checks, both dependency-boundary checks, and the workspace test suite.

The local run produced 1,538 passing tests, 2 failing tests, and 46 ignored tests across 259 test binaries. One failure was a missing `vendor/cfg_defines.h` before `cargo xtask gen-cfg`; the target passes 8 of 8 after the fetch. The remaining failure is `burn-tt --test essentials::dtype_capabilities_advertise_what_the_device_computes_and_no_more`.

Negative evidence is the failing assertion itself: `crates/burn-tt/tests/essentials.rs:23` requires F16 to be storage-only, while `crates/burn-tt/src/lib.rs:217` returns `DTypeUsage::general()` for F16. Commit `cff6742` added F16 arithmetic, so the pinned tree is internally inconsistent.

## Risks / Trade-offs

- The stale test can be fixed upstream in one line. Re-evaluating a later commit may change the verdict; the pin and evidence make that a deliberate act.
- The toolchain repair used for the local run is environment state, not source. It is recorded so the build is not mistaken for hermetic.
- The tt-kmd 2.11.0 ABI pin differs from the host 2.10.0. The observed delta is additive and unused, but a future tree may require the newer ioctl.
- The repository still has no native-Rust model runtime. Reference authority does not close that gap.

## Non-Goals

- Fix the failing upstream test or add a patch queue.
- Add a Nix package, flake input, or pinned Rust toolchain.
- Claim RWKV-7, Qwen3.6/3.8, SmolLM2 parity, or any model support beyond the reviewed source facts.
- Claim physical execution, placement, performance, or silicon conformance.
- Replace TT-Metal, TT-NN, the cxx bridge, or any current runtime.
