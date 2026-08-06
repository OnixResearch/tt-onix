## Why

The repository uses Rust for model validation, command orchestration, and evidence. Direct TT-Metal host execution still crosses C++ process boundaries.

`Knight-Ops/libtt-metal-cxx` exposes a focused TT-Metal host API through the Rust `cxx` bridge. The repository maintainer confirms that it is open-source Tenstorrent work and approves its use here.

A pinned Nix package can establish a reusable Rust-to-Metalium layer. The package needs exact TT-Metal compatibility checks and a narrow validated API before model runtimes depend on it.

## What Changes

- Pin `Knight-Ops/libtt-metal-cxx` at commit `3f62391f9d1cb4d5bdf85266239f63a67f93b016`.
- Bind repository-maintainer approval and record the missing upstream SPDX identifier as metadata debt.
- Add a Nix-owned Rust 2024 and C++20 package against the package-matched TT-Metal output.
- Use the upstream Cargo lockfile without manual edits.
- Add a narrow repository-owned facade for device, unit-mesh, buffer, program, kernel, runtime-argument, enqueue, readback, and close operations.
- Keep validation logic in a pure Rust core and FFI calls in a thin imperative shell.
- Add device-free compile, link, ABI, and negative configuration checks.
- Add one separate bounded Blackhole `1x1` ttsim smoke.
- Add deterministic BLAKE3-bound bridge receipts and explicit non-claims.

## Impact

- **Packages**: new `libtt-metal-cxx` and bridge-validation outputs.
- **Rust and C++**: pinned `cxx` bindings linked to packaged TT-Metal.
- **Validation**: device-free source, build, link, ABI, and facade checks.
- **Simulation**: one optional bounded smoke outside the default gate.
- **Evidence**: one bridge compatibility receipt.

## Out of Scope

- Complete TT-Metal API coverage.
- Profiler, trace, Ethernet, remote circular-buffer, or sub-device APIs.
- Replacing current model runtimes or process adapters.
- Physical hardware execution or performance claims.
- Claiming an exact SPDX identifier before upstream records one.
- Editing `flake.lock` or the upstream Cargo lockfile manually.
