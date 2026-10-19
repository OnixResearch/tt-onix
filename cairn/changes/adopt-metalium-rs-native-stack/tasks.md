# Tasks: Adopt the metalium-rs native stack

## Phase 1: Fork and source authority

- [x] [serial] A1 Fork `Knight-Ops/metalium-rs` to `brittonr/metalium-rs` and record the upstream base `b22a245`. r[tenstorrent.rust_ml.metalium_fork_authority]
- [x] [depends:tenstorrent.rust_ml.metalium_fork_authority] A2 Add the Apache-2.0 license file and fix the stale F16 capability assertion. r[tenstorrent.rust_ml.metalium_fork_authority.valid]
- [x] [depends:tenstorrent.rust_ml.metalium_fork_authority] A3 Run the full upstream validation tier on the fork and record the counts. r[tenstorrent.rust_ml.metalium_fork_authority.valid]
- [x] [serial] A4 Push the fork and pin commit `9afe95d1258667a771ab19a84e589c36eb06c44f`. r[tenstorrent.rust_ml.metalium_fork_authority]

## Phase 2: Toolchain and package

- [x] [depends:tenstorrent.rust_ml.metalium_fork_authority] P1 Add the `rust-overlay` input and pin Rust 1.97.1 with `riscv32im-unknown-none-elf` and `llvm-tools`. r[tenstorrent.rust_ml.metalium_package]
- [x] [depends:tenstorrent.rust_ml.metalium_package] P2 Add the `metalium-rs` package with the pinned fork, Cargo lockfile, and MNIST inputs. r[tenstorrent.rust_ml.metalium_package.valid]
- [x] [depends:tenstorrent.rust_ml.metalium_package] P3 Build the package and confirm `bin/tt-mnist` starts. r[tenstorrent.rust_ml.metalium_package.valid]
- [x] [depends:tenstorrent.rust_ml.metalium_package] P4 Expose the `metalium-rs-mnist` flake app and the package check. r[tenstorrent.rust_ml.metalium_package.valid]
- [x] [depends:tenstorrent.rust_ml.metalium_package] P6 Add `crates/rwkv7` to the fork, expose `metalium-rs-wkv7`, and run the packaged two-token gate on card 0. r[tenstorrent.rust_ml.metalium_migration.recurrence]
- [x] [depends:tenstorrent.rust_ml.metalium_package] P5 Run the full workspace test tier inside the Nix check phase with pinned specification, simulator, and corpus inputs. r[tenstorrent.rust_ml.metalium_package.valid]

## Phase 3: Migration gates

- [ ] [serial] M1 Inventory native operator, state-carry, dtype, and topology coverage for each model service. r[tenstorrent.rust_ml.metalium_migration.incomplete]
- [x] [serial] M2 Run the first single-card physical `tt-mnist --steps 1` gate, then the `1x2` mesh gate. r[tenstorrent.rust_ml.metalium_migration.complete]
- [x] [serial] M3a Implement the WKV7 host reference and composed Burn device path; verify output and two-token state carry on card 0. r[tenstorrent.rust_ml.metalium_migration.recurrence]
- [x] [serial] M3b Implement one layer from the pinned equations with synthetic weights; compare Flex FP32 with TtBackend across two tokens on card 0. r[tenstorrent.rust_ml.metalium_migration.layer]
- [x] [serial] M3c Load the real BF16 checkpoint for layer zero, including rank-128 g-LoRA and per-head `r_k`; compare Flex with TtBackend on card 0 after `tt-smi -r`. r[tenstorrent.rust_ml.metalium_migration.real_layer]
- [x] [serial] M3d Load all 12 layers including rank-32 v-LoRA; compare the final normalized hidden and every recurrent matrix between Flex and TtBackend on card 0. r[tenstorrent.rust_ml.metalium_migration.model]
- [x] [serial] M3e Evaluate logits and compare final hidden, logits, and recurrent states against the harness Python reference on card 0. r[tenstorrent.rust_ml.metalium_migration.logits]
- [ ] [serial] M3 Blocked: Fuse the WKV7 recurrence into a Tensix kernel and re-verify against the host reference. r[tenstorrent.rust_ml.metalium_migration.complete]
- [ ] [serial] M4 Blocked: Implement and verify Qwen3.6/3.8 linear attention, MoE routing, and state carry on `burn-tt`. r[tenstorrent.rust_ml.metalium_migration.complete]
- [ ] [serial] M5 Blocked: Admit each model only after its parity and physical gates pass. r[tenstorrent.rust_ml.metalium_migration.complete]

- [ ] [serial] L1 Blocked: resolve LGPL-2.1-only repository terms against the Apache-2.0 fork before linking its crates into repository binaries. r[tenstorrent.rust_ml.metalium_adoption_evidence]

## Phase 4: Records and validation

- [x] [serial] D1 Update `AGENTS.md` and `README.md` with the fork, package, migration policy, and non-claims. r[tenstorrent.rust_ml.metalium_adoption_evidence]
- [x] [depends:tenstorrent.rust_ml.metalium_adoption_evidence] E1 Write the adoption evidence with source identities, counts, package output, and binary smoke. r[tenstorrent.rust_ml.metalium_adoption_evidence.valid]
- [x] [depends:tenstorrent.rust_ml.metalium_adoption_evidence] V1 Run Cairn validation and the proposal, design, and tasks gates. r[tenstorrent.rust_ml.metalium_adoption_evidence.valid]
- [x] [serial] S1 Mark the cxx bridge plan superseded for native device work without deleting its record. r[tenstorrent.rust_ml.metalium_fork_authority]

## Verification Coverage

- `Scenario: Fork revision is admitted` -> A1, A2, A4
- `Scenario: Fork or upstream base drifts` -> A1, A4
- `Scenario: Package builds and installs the binary` -> P1, P2, P3
- `Scenario: Toolchain or input is unpinned` -> P1, P2
- `Scenario: Model parity is incomplete` -> M1
- `Scenario: WKV7 recurrence and state carry pass` -> M3a, P6
- `Scenario: Synthetic layer matches across backends` -> M3b
- `Scenario: Real layer zero matches across backends` -> M3c
- `Scenario: All 12 real layers match across backends` -> M3d
- `Scenario: Logits match the Python reference` -> M3e
- `Scenario: Model gates pass` -> M2, M3, M4, M5
- `Scenario: Adoption evidence is recorded` -> D1, E1, V1
- `Scenario: License scope is unresolved` -> L1
- `Scenario: Build evidence is presented as model support` -> E1
