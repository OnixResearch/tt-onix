## Why

`Knight-Ops/metalium-rs` is a native Rust Blackhole stack: bare-metal `riscv32im` firmware, a `tt-kmd` host transport, Tensix kernels, and the `burn-tt` backend. Shipped crates have no TT-Metalium, TT-NN, simulator, or C/C++ dependency. It is the only reviewed source that covers host access, firmware, kernels, and a Burn backend without the TT-Metalium C++ stack.

It is not a `cxx` bridge. It does not expose the TT-Metal host API, so it cannot satisfy the `package-libtt-metal-cxx` change. It also does not implement a model runtime, RWKV-7, Qwen, MoE, or gated-delta.

The repository needs one exact, reviewed revision and a local validation result before it treats this tree as authority for native Rust device, firmware, kernel, or Burn-backend work.

## What Changes

- Pin `Knight-Ops/metalium-rs` commit `b22a245db18252b131679aca992dac85c9f7d812` as the reviewed native-Rust implementation reference.
- Record the source license facts: the workspace manifest declares Apache-2.0; the pinned tree has no separate license file.
- Record the pinned inputs: ttsim `v1.10.9`, tt-kmd `ttkmd-2.11.0` IOCTL API version 2, tt-metal `cfg_defines.h` at `81989dcdb8f9b340c932ae7a71a346f4f08703eb`, and Burn `=0.21.0`.
- Record a local build and validation result with exact commands.
- Record the blocker: the pinned revision fails its own `burn-tt --test essentials` capability assertion.
- Record the Nix toolchain gap: the nixpkgs Rust dev shell lacks the `riscv32im-unknown-none-elf` standard library and `llvm-objcopy`/`llvm-objdump`.
- Keep `package-libtt-metal-cxx` and `establish-rust-ml-runtime-boundary` unchanged; this change adds a reference authority and a blocked package verdict, not a replacement bridge.

## Impact

- **References**: the pinned checkout stays at `.pi/references/tenstorrent/metalium-rs`; `AGENTS.md` and `README.md` name it.
- **Validation**: one device-free build and test evaluation, including simulator gates.
- **Evidence**: `evidence/evaluation.md` records the environment, commands, counts, and exact blocker.
- **Packages**: none. No Nix package or flake input is added.
- **Model runtimes**: none.

## Out of Scope

- A Nix package, flake input, or pinned Rust toolchain.
- Patching the upstream test or any other source file.
- Model runtime adapters, RWKV-7, Qwen3.6/3.8, MoE, or gated-delta support.
- Physical hardware execution, placement, performance, or conformance claims.
- Replacing the `libtt-metal-cxx` bridge or the packaged TT-Metal/ttnn paths.
