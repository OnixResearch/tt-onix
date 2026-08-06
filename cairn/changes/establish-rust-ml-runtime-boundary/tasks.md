# Tasks: Establish the Rust ML runtime boundary

## Phase 1: Baseline and contract inventory

- [ ] [serial] B1 Record current chat, P150x2 command, dispatch ABI, focused package, and Cairn validation results before edits. r[tenstorrent.rust_ml.runtime_contract]
- [ ] [serial] C1 Inventory current load, prefill, decode, state-carry, reset, close, timeout, topology, and evidence behavior. r[tenstorrent.rust_ml.runtime_contract]
- [ ] [depends:tenstorrent.rust_ml.runtime_contract] C2 Author the typed Nickel runtime registry and generate its canonical projection. r[tenstorrent.rust_ml.runtime_contract]

## Phase 2: Functional core and adapters

- [ ] [depends:tenstorrent.rust_ml.runtime_contract] R1 Add the pure Rust capability, request, response, session-transition, error, and receipt core. r[tenstorrent.rust_ml.adapter_boundary]
- [ ] [depends:tenstorrent.rust_ml.adapter_boundary] A1 Add a thin `llama-cpp-metalium` adapter that preserves the current `rwkv7-chat` defaults. r[tenstorrent.rust_ml.adapter_boundary]
- [ ] [depends:tenstorrent.rust_ml.adapter_boundary] A2 Add a thin persistent P150x2 adapter that reuses the current bounded dispatch core. r[tenstorrent.rust_ml.adapter_boundary]
- [ ] [depends:tenstorrent.rust_ml.adapter_boundary] A3 Add a deterministic in-memory fake adapter for device-free success and failure tests. r[tenstorrent.rust_ml.adapter_boundary]

## Phase 3: Source admission and evidence

- [ ] [serial] S1 Add a typed third-party runtime-source admission manifest with URL, revision, content identity, license, API scope, and verdict. r[tenstorrent.rust_ml.source_admission]
- [ ] [depends:tenstorrent.rust_ml.source_admission] S2 Record `libtt-metal-cxx` commit `3f62391f9d1cb4d5bdf85266239f63a67f93b016` as rejected because the reviewed source lacks an explicit license. r[tenstorrent.rust_ml.source_admission]
- [ ] [depends:tenstorrent.rust_ml.adapter_boundary] E1 Add the canonical BLAKE3-bound runtime receipt and execution classification. r[tenstorrent.rust_ml.runtime_evidence]

## Phase 4: Verification and documentation

- [ ] [parallel] V1 Add positive tests for valid registry export, capability admission, complete fake sessions, current adapter construction, and deterministic receipts. r[tenstorrent.rust_ml.runtime_contract] r[tenstorrent.rust_ml.runtime_evidence]
- [ ] [parallel] V2 Add negative tests for unsupported operations, invalid transitions, malformed output, timeout, global process termination, missing licenses, and false physical claims. r[tenstorrent.rust_ml.source_admission] r[tenstorrent.rust_ml.runtime_evidence]
- [ ] [serial] D1 Document backend selection, exact capabilities, source admission, receipt interpretation, and hardware non-claims. r[tenstorrent.rust_ml.runtime_evidence]
- [ ] [depends:tenstorrent.rust_ml.runtime_evidence] V3 Run formatting, Cargo tests, focused Nix checks, existing runtime regressions, Cairn validation, and all lifecycle gates. r[tenstorrent.rust_ml.runtime_evidence]

## Verification Coverage

- `Scenario: Valid runtime profile exports canonical data` -> C1, C2, V1
- `Scenario: Invalid capability declaration fails` -> C2, V2
- `Scenario: Complete fake session succeeds` -> R1, A3, V1
- `Scenario: Invalid session transition fails` -> R1, A3, V2
- `Scenario: Current adapters preserve command behavior` -> A1, A2, V1
- `Scenario: Adapter cannot terminate an unrelated process` -> A1, A2, V2
- `Scenario: Licensed source is admitted` -> S1, V1
- `Scenario: Source without a license is rejected` -> S1, S2, V2
- `Scenario: Device-free receipt is deterministic` -> E1, V1
- `Scenario: False physical classification is rejected` -> E1, V2
