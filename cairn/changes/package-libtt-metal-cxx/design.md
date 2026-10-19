> Status: superseded for native device work by `adopt-metalium-rs-native-stack`.
> Keep this change as a reviewed record; do not implement the cxx bridge.

## Context

`libtt-metal-cxx` uses `cxx::bridge` to connect Rust with TT-Metal C++ APIs. Its reviewed tree includes device, program, kernel, buffer, circular-buffer, semaphore, runtime-argument, tilize, and unit-mesh surfaces.

The upstream `build.rs` already accepts `TT_METAL_INSTALL_DIR`, `TT_METAL_INCLUDE_DIR`, `TT_METAL_LIB_DIR`, and `TT_METAL_RUNTIME_ROOT`. It links `fmt`, `tt_stl`, and `tt_metal` and requires C++20.

The reviewed source has a Cargo lockfile. It has no license file or Cargo license field. The repository maintainer approves its use as open-source Tenstorrent work. The exact SPDX identifier remains unknown.

## Decisions

### Decision: Pin source, authorization, and lock identity

**Choice:** Pin URL and commit `3f62391f9d1cb4d5bdf85266239f63a67f93b016`. Bind the Nix source hash, BLAKE3 tree identity, Cargo lock identity, authorization basis, and license status.

Record `repository_maintainer_approval` as the authorization basis. Record the SPDX identifier as unknown instead of assigning a guessed license.

**Rationale:** The exact source and dependency graph must remain reproducible. Authorization and SPDX metadata are separate facts.

### Decision: Build only against packaged TT-Metal

**Choice:** Use `rustPlatform.buildRustPackage` with the pinned Cargo lockfile and Nix-owned Rust and C++ tools. Pass only store paths through the four supported TT-Metal build variables.

Do not use `/usr/local`, a sibling checkout, ambient Cargo configuration, or a user runtime root. Do not modify `flake.lock` or the Cargo lockfile by hand.

**Rationale:** The package must match this repository's TT-Metal API and runtime layout.

### Decision: Expose one bounded facade

**Choice:** Add a repository-owned facade over the admitted upstream crate. The first facade covers device query, unit mesh, buffers, programs, inline kernels, runtime arguments, enqueue, readback, and close.

The facade does not expose unchecked raw pointers or unbounded byte counts. Closed handles and invalid dimensions return typed errors instead of panicking.

**Rationale:** The upstream crate is a low-level bridge. Model code needs a smaller contract with repository-owned limits and errors.

### Decision: Use a functional core and an imperative FFI shell

**Choice:** Pure functions validate dimensions, byte counts, core ranges, operation order, and expected result sizes. They return an execution plan.

The shell converts an admitted plan into `libtt-metal-cxx` calls. It owns handles, cleanup, and C++ exception conversion. It closes resources in reverse acquisition order.

**Rationale:** Core validation can run without TT-Metal, ttsim, or silicon. The unsafe execution surface stays small.

### Decision: Fail on package-matched API drift

**Choice:** Add compile and link fixtures for every facade operation. Bind expected TT-Metal headers, shared libraries, runtime root, and symbols to one compatibility receipt.

A missing header, symbol, library, or runtime directory blocks the package. The change does not update TT-Metal to fit the bridge.

**Rationale:** Upstream bridge support can target a newer TT-Metal API. A precise mismatch is safer than a hidden compatibility patch.

### Decision: Keep simulator and hardware gates separate

**Choice:** Default checks build, link, inspect symbols, and run pure positive and negative tests. They cannot open `/dev/tenstorrent`, query hardware, or load ttsim.

A separate Blackhole `1x1` ttsim smoke opens a unit mesh, allocates one bounded buffer, runs one inline kernel, reads the result, and closes cleanly. It uses slow dispatch and a named timeout.

Physical hardware remains a later explicit gate.

**Rationale:** Fast checks protect packaging and API compatibility. Simulator evidence proves only the bounded simulated path.

### Decision: Emit a bridge compatibility receipt

**Choice:** The canonical receipt binds source, authorization, lockfile, Rust toolchain, C++ toolchain, TT-Metal identity, API subset, checks, execution class, outcome, and exact blockers. BLAKE3 binds canonical bytes.

**Rationale:** Runtime changes need one stable input that proves which bridge and TT-Metal pair passed.

## Validation Strategy

Positive device-free tests cover source admission, lock use, Nix path resolution, build, link, symbol presence, valid plans, error conversion, deterministic cleanup plans, and receipt stability.

Negative tests cover source drift, lock drift, ambient paths, missing headers, missing symbols, invalid core ranges, zero or oversized buffers, result-size mismatch, use after close, partial acquisition, and false hardware claims.

The separate simulator smoke covers one unit mesh and one inline-kernel round trip. It records a simulated receipt and no performance result.

## Risks / Trade-offs

- The approved source lacks an exact SPDX identifier. The package records this debt and does not invent license terms.
- The bridge can lag packaged TT-Metal. Compatibility checks return the exact blocked API.
- The narrow facade omits useful APIs. Later changes can add them with positive and negative tests.
- C++ exceptions cross an FFI boundary. The shell converts them into typed Rust errors.
- Simulator success cannot prove silicon behavior, placement, or performance.

## Non-Goals

- Copy the full TT-Metal API into Rust.
- Add model, tensor-framework, or serving behavior.
- Run default checks on a device or simulator.
- Update TT-Metal, firmware, driver, or simulator pins.
