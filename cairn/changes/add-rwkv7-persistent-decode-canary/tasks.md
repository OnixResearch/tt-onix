# Tasks

## Phase 1: Policy and interlock

- [x] [serial] Add the strict Nickel canary policy, throughput profile, and invalid fixtures. r[tenstorrent.rwkv7_p150x2.persistent_decode_canary.policy]
- [x] [serial] Add pure C++ interlock validation and force normal dispatch for blocked states. r[tenstorrent.rwkv7_p150x2.persistent_decode_canary.interlock]
- [x] [serial] Add runtime shell loading for exact interlock path, file BLAKE3, canary binding, and profile identity. r[tenstorrent.rwkv7_p150x2.persistent_decode_canary.runtime]

## Phase 2: Canary controller

- [x] [serial] Add the pure Rust budget, A/B, monitoring, decision, and receipt core. r[tenstorrent.rwkv7_p150x2.persistent_decode_canary.evaluate]
- [x] [serial] Add the bounded Rust command with atomic interlocks, fixed paths, named timeout, and stable statuses. r[tenstorrent.rwkv7_p150x2.persistent_decode_canary.command]
- [x] [serial] Add positive and negative C++, Rust, Nickel, and package tests. r[tenstorrent.rwkv7_p150x2.persistent_decode_canary.invalid]
- [x] [serial] Document the operator contract, interlock environment, statuses, and non-claims. r[tenstorrent.rwkv7_p150x2.persistent_decode_canary.command]

## Phase 3: Physical canary and completion

- [x] [serial] Run one bounded physical canary and retain A/B, monitoring, interlock, and canary receipts. r[tenstorrent.rwkv7_p150x2.persistent_decode_canary.physical]
- [ ] [serial] Run runtime, evidence, formatting, source-boundary, closure, repository-quality, and Cairn gates. r[tenstorrent.rwkv7_p150x2.persistent_decode_canary.invalid]
- [ ] [serial] Sync the accepted requirement, archive the change, regenerate the artifact manifest, and integrate the verified branch. r[tenstorrent.rwkv7_p150x2.persistent_decode_canary.runtime]
