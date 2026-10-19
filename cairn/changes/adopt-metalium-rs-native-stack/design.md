## Context

`brittonr/metalium-rs` at commit `8600f145abfc9955fa1238b2f2e9f51d34b3b3d8` is a fork of `Knight-Ops/metalium-rs` at `b22a245`. The fork adds `LICENSE` (Apache-2.0), fixes one test assertion, and adds `crates/rwkv7`, a host reference plus composed device gate for the WKV7 recurrence.

The full fork validation tier passes: 1,540 tests, 0 failures, 46 ignored, 259 binaries. The Nix package builds the shipped crates and `tt-mnist`.

The repository's Rust code is host-side validation and evidence today. Device work still runs through TT-Metal/ttnn, llama.cpp Metalium, or the ttWKV7 runtime. `metalium-rs` has no RWKV-7, Qwen, MoE, or gated-delta implementation, so no model service can switch yet.

## Decisions

### Decision: The fork is the package authority; upstream remains the comparison base

**Choice:** Pin the fork commit and record the exact delta from upstream. Keep upstream as the reference for behavior the fork did not change.

Update the fork by rebasing the reviewed changes onto a reviewed upstream commit. Do not merge unrelated upstream work into the pinned package without a new validation run.

**Rationale:** The fork needs a license file and the test fix to pass its own tier. A recorded delta keeps future upstream refreshes auditable.

### Decision: Pin the Rust toolchain, not the ambient rustup install

**Choice:** Add `rust-overlay` and pin Rust 1.97.1 with `riscv32im-unknown-none-elf` and `llvm-tools`.

The package uses `makeRustPlatform` with that toolchain, so `rustc`, `cargo`, the firmware target, and `llvm-objcopy`/`llvm-objdump` all resolve from Nix inputs. Do not reuse the host rustup installation.

**Rationale:** The nixpkgs development shell lacks the firmware target and LLVM tools. An ambient rustup toolchain makes the package irreproducible.

### Decision: Build the shipped stack and the MNIST binary

**Choice:** Build the workspace with only dev-only crates excluded. Provide the pinned MNIST corpus and install `bin/tt-mnist`.

The excluded crates need the ISA specification tree, ttsim payloads, or model weights. Keep them in the source validation lane, not the package.

**Rationale:** An empty library-only derivation proves compilation but gives no runnable artifact. `tt-mnist` is self-contained and exercises firmware, kernels, device transport, and the Burn backend.

### Decision: Start WKV7 as a composed gate, then fuse

**Choice:** Add `crates/rwkv7` with the pure F32 host reference and the same equation composed from
Burn tensor operations on `TtBackend`. Use `accurate_matmul` and verify two-token state carry on
card 0.

Do not call this a fused kernel or a model. The fused Tensix recurrence remains the next gate.

**Rationale:** The composed path proves the equation, the state contract, and the existing backend
ops before a new kernel exists. It gives a reference the fused kernel can be compared against.

### Decision: Migrate by parity gates, not by service replacement

**Choice:** Route new Rust device, firmware, kernel, and Burn-backend work to `metalium-rs`.

Keep each existing model service on its current runtime until the native stack passes that model's gates: operator coverage, recurrent-state carry, numerical agreement, mesh topology, and physical evidence.

**Rationale:** The native stack has no model parity today. Replacing a serving path on build success would convert partial evidence into a support claim.

### Decision: Keep the package separate until license compatibility is decided

**Choice:** Distribute `metalium-rs` as a separate Nix package. Do not link its crates into
repository-authored binaries until the owner records a license decision.

The repository declares LGPL-2.1-only for its own code. The fork is Apache-2.0. A package
derivation aggregates the work; a linked binary is a combined work with different obligations.
The package build and physical smoke do not settle that question.

**Rationale:** License scope is a distribution fact, not an engineering inference. The native
stack can be evaluated and run without pretending the combined-work terms are resolved.

### Decision: Supersede the cxx bridge for native device work

**Choice:** Do not implement `package-libtt-metal-cxx`. Keep its proposal and design as a reviewed record. `metalium-rs` does not expose TT-Metal host APIs, so the two are not interchangeable.

**Rationale:** The migration direction is native Rust. Maintaining a second Rust-to-C++ bridge adds a toolchain and ABI surface the native stack removes.

## Validation Strategy

Positive checks cover the fork pin, the license file, the F16 test fix, the WKV7 host reference, the full validation tier, a hermetic package build, the installed `tt-mnist` and `wkv7-gate` binaries, and the WKV7 two-token state-carry gate on card 0.

Negative checks cover source drift, an unpinned toolchain, a missing firmware target, a missing license, and a package that claims a model runtime. The installed output excludes dev-only crates; the check phase runs their simulator-backed tier, and no check opens hardware.

Physical validation is separate. The first hardware gate is `tt-mnist --steps 1 --card 0` on one card; mesh work follows only after the single-card path is measured.

## Risks / Trade-offs

- The fork can drift from upstream. The recorded delta and a revalidation requirement keep it reviewable.
- The composed WKV7 gate is correctness evidence only; it is neither a fused-kernel cost nor a layer/model result.
- Rust 1.97.1 is pinned through `rust-overlay`; a future lock update can change the overlay revision even if the toolchain version stays fixed.
- `tt-mnist` embeds about 11 MB of compressed data and adds build time. It is the only self-contained native binary today.
- Excluding dev-only crates leaves the conformance and simulator gates outside the package check.
- Model migration can stall on missing kernels. That is an explicit blocker, not a packaging defect.
- LGPL-2.1-only versus Apache-2.0 combined-work terms remain unresolved; keep the package separate
  until the owner decides.

## Non-Goals

- Claim Qwen3.6/3.8, RWKV-7, MoE, or gated-delta support on `metalium-rs`.
- Run hardware or simulator workloads in the default flake check.
- Remove TT-Metal, TT-NN, ttsim, or the current model runtimes.
- Track upstream automatically or publish the fork.
- Implement the cxx bridge.
