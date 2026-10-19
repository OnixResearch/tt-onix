## Why

`metalium-rs` is the only reviewed source that covers Blackhole host access, bare-metal firmware, Tensix kernels, and a Burn backend without TT-Metalium, TT-NN, Python, or C++. The repository already depends on it as the native Rust reference.

The upstream revision cannot be the package authority as-is: it has no license file and fails one stale test. The fork fixes both. The repository now needs a pinned fork, a reproducible Nix package and toolchain, and an explicit migration policy.

## What Changes

- Pin the fork `brittonr/metalium-rs` at commit `9afe95d1258667a771ab19a84e589c36eb06c44f`, based on upstream `b22a245`.
- Record the fork delta: the Apache-2.0 license file, the F16 capability assertion fix, and the `crates/rwkv7` WKV7 reference/device gate.
- Add the `rust-overlay` flake input and a Rust 1.97.1 toolchain with `riscv32im-unknown-none-elf` and `llvm-tools`.
- Add the `metalium-rs` Nix package. It builds the shipped stack and the self-contained `tt-mnist` binary, and excludes only dev-only crates.
- Add the `metalium-rs-mnist` and `metalium-rs-wkv7` flake apps.
- Record the first native WKV7 state-carry gate, a synthetic full-layer Flex/Tt gate, a real-checkpoint layer-zero gate, and an all-12-layer final-hidden/logits/recurrent-state gate on card 0.
- Generate the Python equation-reference artifacts and compare the device logits and states against them.
- Record the board reset required after a TT-Metal service stop.
- Record the phased migration policy: new Rust device, firmware, kernel, and Burn-backend work uses this stack; existing Qwen3.6/3.8 and RWKV-7 services stay on TT-Metal/ttnn until model parity is proven.
- Mark the `libtt-metal-cxx` bridge plan superseded for native device work. It remains a record, not an implementation target.

## Impact

- **Flake**: one new input (`rust-overlay`) and lock entry; one package, one check, one app.
- **Toolchain**: pinned Rust 1.97.1 plus the firmware target and LLVM tools.
- **Packages**: `metalium-rs` with `bin/tt-mnist`.
- **Documentation**: `AGENTS.md`, `README.md`, and Cairn records.
- **Model runtimes**: none switched in this change.

## Out of Scope

- Porting Qwen3.6/3.8, RWKV-7, MoE, gated-delta, or any model runtime.
- Physical hardware execution, firmware flashing, placement, performance, or conformance claims.
- Keeping the fork in lockstep with upstream automatically.
- Removing the TT-Metal/ttnn packages or services.
- Implementing the `libtt-metal-cxx` bridge.
- Resolving LGPL-2.1-only versus Apache-2.0 combined-work terms.
