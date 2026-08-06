# Tasks: Package libtt-metal-cxx

## Phase 1: Baseline and authority

- [ ] [serial] B1 Record TT-Metal package identities, Rust and C++ toolchains, bridge source facts, focused checks, and Cairn validation before edits. r[tenstorrent.rust_ml.libtt_authority]
- [ ] [serial] A1 Pin `Knight-Ops/libtt-metal-cxx` at commit `3f62391f9d1cb4d5bdf85266239f63a67f93b016`. Bind Nix and BLAKE3 source identities and Cargo lock identity. r[tenstorrent.rust_ml.libtt_authority]
- [ ] [depends:tenstorrent.rust_ml.libtt_authority] A2 Bind repository-maintainer approval and record the unknown SPDX identifier as metadata debt. r[tenstorrent.rust_ml.libtt_authority]

## Phase 2: Nix package and compatibility

- [ ] [depends:tenstorrent.rust_ml.libtt_authority] P1 Add the `libtt-metal-cxx` package with Nix-owned Rust 2024, C++20, `cxx`, and the upstream Cargo lockfile. r[tenstorrent.rust_ml.libtt_package]
- [ ] [depends:tenstorrent.rust_ml.libtt_package] P2 Pass packaged TT-Metal include, library, install, and runtime roots through explicit build variables. r[tenstorrent.rust_ml.libtt_package]
- [ ] [depends:tenstorrent.rust_ml.libtt_package] P3 Add compile, link, runtime-root, and required-symbol compatibility fixtures. r[tenstorrent.rust_ml.libtt_package]
- [ ] [depends:tenstorrent.rust_ml.libtt_package] P4 Add negative fixtures for source drift, lock drift, ambient paths, missing headers, missing libraries, and missing symbols. r[tenstorrent.rust_ml.libtt_package]

## Phase 3: Bounded Rust facade

- [ ] [depends:tenstorrent.rust_ml.libtt_package] F1 Add pure validation types for dimensions, byte counts, core ranges, operation order, result sizes, and cleanup plans. r[tenstorrent.rust_ml.libtt_facade]
- [ ] [depends:tenstorrent.rust_ml.libtt_facade] F2 Add the thin FFI shell for device query, unit mesh, buffers, programs, inline kernels, runtime arguments, enqueue, readback, and close. r[tenstorrent.rust_ml.libtt_facade]
- [ ] [depends:tenstorrent.rust_ml.libtt_facade] F3 Convert C++ exceptions and closed handles into typed Rust errors. Close partially acquired resources in reverse order. r[tenstorrent.rust_ml.libtt_facade]
- [ ] [parallel] F4 Add positive and negative core tests for valid plans, invalid ranges, zero and oversized buffers, result mismatch, use after close, and partial acquisition. r[tenstorrent.rust_ml.libtt_facade]

## Phase 4: Separate simulator smoke and evidence

- [ ] [depends:tenstorrent.rust_ml.libtt_facade] S1 Add a separate Blackhole `1x1` ttsim smoke with slow dispatch and a named timeout. r[tenstorrent.rust_ml.libtt_evidence]
- [ ] [depends:tenstorrent.rust_ml.libtt_evidence] S2 Run one unit-mesh, bounded-buffer, inline-kernel, readback, and close round trip. Emit a simulated receipt. r[tenstorrent.rust_ml.libtt_evidence]
- [ ] [depends:tenstorrent.rust_ml.libtt_package] E1 Add the canonical BLAKE3-bound bridge compatibility receipt with exact blockers and non-claims. r[tenstorrent.rust_ml.libtt_evidence]

## Phase 5: Final validation

- [ ] [parallel] V1 Prove default checks cannot open a device, load ttsim, or infer physical execution. r[tenstorrent.rust_ml.libtt_evidence]
- [ ] [serial] D1 Document the source pin, approval, SPDX metadata debt, TT-Metal compatibility, facade scope, simulator command, and non-claims. r[tenstorrent.rust_ml.libtt_evidence]
- [ ] [depends:tenstorrent.rust_ml.libtt_evidence] V2 Run formatting, Cargo tests, package build, focused Nix checks, ABI fixtures, Cairn validation, and all lifecycle gates. Run ttsim only through its separate check. r[tenstorrent.rust_ml.libtt_evidence]

## Verification Coverage

- `Scenario: Approved bridge source is admitted` -> A1, A2
- `Scenario: Bridge source or lock drifts` -> A1, P4
- `Scenario: Package builds against packaged TT-Metal` -> P1, P2, P3
- `Scenario: Required TT-Metal API is missing` -> P3, P4
- `Scenario: Valid facade plan executes in order` -> F1, F2, F4
- `Scenario: Invalid facade input fails before FFI` -> F1, F4
- `Scenario: Partial acquisition cleans up safely` -> F2, F3, F4
- `Scenario: Default validation stays device-free` -> P3, V1
- `Scenario: Simulator round trip is classified as simulated` -> S1, S2
- `Scenario: Simulator receipt claims physical execution` -> E1, V1
- `Scenario: Compatibility receipt is deterministic` -> E1, V2
