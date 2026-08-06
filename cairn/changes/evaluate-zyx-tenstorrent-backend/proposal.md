## Why

`zyx` is the reviewed Rust ML framework with the most direct Tenstorrent integration. It contains a native backend, Tenstorrent code generation, and TT-Metal kernel generation.

The backend remains experimental. Upstream states that the full test suite does not pass. The reviewed source also has incomplete operator paths, broad capability claims, user-directory build output, and global process termination.

The repository needs a bounded evaluation before it uses `zyx` as a model runtime or support dependency. A failed evaluation with exact blockers is a valid result.

## What Changes

- Pin `zk4x/zyx` version `0.16.0` at commit `7606188adac64b362c257159af9949d860759cc0` with its LGPL-3.0-only license.
- Add a Nix-owned experimental package for the Tenstorrent feature.
- Patch the C++ runtime build into Cargo output and install it under the package output.
- Use packaged TT-Metal headers, libraries, and runtime data instead of checkout and user-directory paths.
- Replace global runtime termination with exact child ownership and cleanup.
- Add a reviewed dtype and operator capability matrix.
- Add device-free compiler and code-generation fixtures with positive and negative cases.
- Add one optional bounded Blackhole `1x1` ttsim smoke outside the default device-free gate.
- Produce a deterministic evaluation verdict with exact blockers and non-claims.

## Impact

- **Packages**: one experimental `zyx-tenstorrent` package and source pin.
- **Rust and C++**: a local patch queue for build output, runtime discovery, process ownership, and capability reporting.
- **Validation**: device-free code-generation checks and one separate simulator smoke.
- **Evidence**: a BLAKE3-bound evaluation receipt.

## Out of Scope

- A support claim for `zyx`, RWKV-7, Qwen3.6, or any complete model.
- Fixing the complete upstream test suite.
- Physical hardware execution or performance measurement.
- Long simulator inference.
- A Burn, Candle, CubeCL, or `mistral.rs` backend.
- Updating the packaged TT-Metal pin to match upstream `zyx`.
