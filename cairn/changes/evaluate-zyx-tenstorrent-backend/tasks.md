# Tasks: Evaluate the zyx Tenstorrent backend

## Phase 1: Baseline and source authority

- [ ] [serial] B1 Record current Rust, TT-Metal, ttsim, flake, and Cairn validation results before edits. r[tenstorrent.rust_ml.zyx_authority]
- [ ] [serial] A1 Pin `zk4x/zyx` version `0.16.0` at commit `7606188adac64b362c257159af9949d860759cc0`. Bind Nix and BLAKE3 source identities, Cargo lock identity, and LGPL-3.0-only license. r[tenstorrent.rust_ml.zyx_authority]
- [ ] [depends:tenstorrent.rust_ml.zyx_authority] A2 Add source-drift, lock-drift, and missing-license negative fixtures. r[tenstorrent.rust_ml.zyx_authority]

## Phase 2: Hermetic package and lifecycle

- [ ] [depends:tenstorrent.rust_ml.zyx_authority] P1 Add the experimental `zyx-tenstorrent` package with the Nix-owned Rust and C++ toolchain. r[tenstorrent.rust_ml.zyx_hermetic_runtime]
- [ ] [depends:tenstorrent.rust_ml.zyx_hermetic_runtime] P2 Patch runtime compilation into Cargo `OUT_DIR` and install it under `$out/libexec/zyx`. r[tenstorrent.rust_ml.zyx_hermetic_runtime]
- [ ] [depends:tenstorrent.rust_ml.zyx_hermetic_runtime] P3 Patch TT-Metal discovery to use explicit packaged include, library, and runtime roots. r[tenstorrent.rust_ml.zyx_hermetic_runtime]
- [ ] [depends:tenstorrent.rust_ml.zyx_hermetic_runtime] P4 Replace global process termination with exact child termination and reaping. Extract protocol and lifecycle decisions into pure functions. r[tenstorrent.rust_ml.zyx_hermetic_runtime]

## Phase 3: Capability matrix and device-free checks

- [ ] [serial] C1 Inventory allocation, transfer, unary, binary, movement, reduction, matmul, realization, dtype, and architecture paths. r[tenstorrent.rust_ml.zyx_capability_matrix]
- [ ] [depends:tenstorrent.rust_ml.zyx_capability_matrix] C2 Author the typed capability matrix and bind each supported entry to one fixture. r[tenstorrent.rust_ml.zyx_capability_matrix]
- [ ] [depends:tenstorrent.rust_ml.zyx_capability_matrix] C3 Add positive CPU, graph, lowering, and Tenstorrent code-generation fixtures. r[tenstorrent.rust_ml.zyx_capability_matrix]
- [ ] [depends:tenstorrent.rust_ml.zyx_capability_matrix] C4 Add negative fixtures for unsupported dtypes, incomplete operations, malformed runtime messages, and unsupported architecture claims. r[tenstorrent.rust_ml.zyx_capability_matrix]

## Phase 4: Bounded simulator evaluation

- [ ] [depends:tenstorrent.rust_ml.zyx_hermetic_runtime] S1 Add a separate Blackhole `1x1` ttsim smoke with slow dispatch and a named timeout. Keep it outside the default device-free gate. r[tenstorrent.rust_ml.zyx_evaluation_verdict]
- [ ] [depends:tenstorrent.rust_ml.zyx_evaluation_verdict] S2 Record runtime startup, transfer, one supported operation, readback, cleanup, and exact blockers in a simulated receipt. r[tenstorrent.rust_ml.zyx_evaluation_verdict]

## Phase 5: Verdict and final validation

- [ ] [depends:tenstorrent.rust_ml.zyx_capability_matrix] E1 Produce a canonical BLAKE3-bound `blocked` or `eligible_for_runtime_adapter_spike` verdict. r[tenstorrent.rust_ml.zyx_evaluation_verdict]
- [ ] [parallel] V1 Add negative package checks for home writes, checkout paths, global process termination, and false physical classification. r[tenstorrent.rust_ml.zyx_hermetic_runtime] r[tenstorrent.rust_ml.zyx_evaluation_verdict]
- [ ] [serial] D1 Document the source pin, local patches, capability matrix, simulator command, verdict, LGPL metadata, and explicit non-claims. r[tenstorrent.rust_ml.zyx_evaluation_verdict]
- [ ] [depends:tenstorrent.rust_ml.zyx_evaluation_verdict] V2 Run formatting, Cargo tests, focused Nix checks, package build, device-free fixtures, Cairn validation, and all lifecycle gates. Run the simulator smoke only through its separate check. r[tenstorrent.rust_ml.zyx_evaluation_verdict]

## Verification Coverage

- `Scenario: Pinned zyx source is admitted` -> A1
- `Scenario: Source or lock drift is rejected` -> A2
- `Scenario: Hermetic package installs its runtime` -> P1, P2, P3
- `Scenario: Build tries to write a user directory` -> P2, V1
- `Scenario: Runtime closes its owned child` -> P4
- `Scenario: Runtime tries global termination` -> P4, V1
- `Scenario: Supported operation has matching fixtures` -> C1, C2, C3
- `Scenario: Unsupported capability fails before launch` -> C2, C4
- `Scenario: Bounded simulator smoke is classified as simulated` -> S1, S2
- `Scenario: Simulator evidence is presented as physical` -> S2, V1
- `Scenario: Passing gates permit only an adapter spike` -> E1
- `Scenario: Missing gate produces an exact blocker` -> E1
