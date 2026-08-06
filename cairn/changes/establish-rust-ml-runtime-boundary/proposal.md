## Why

The repository has several model execution paths, but each command binds directly to one runtime. `rwkv7-chat` invokes `llama-cli`. The persistent RWKV-7 path uses its own command and dispatch protocol.

This coupling makes runtime comparison difficult. It also lets backend-specific process rules, capability claims, and evidence formats enter user commands.

A small Rust runtime boundary can reuse the current pure dispatch logic. It can keep process and device work in thin adapters. It can also admit future Rust runtimes without changing chat or evidence consumers.

## What Changes

- Add a typed Nickel contract for runtime identity, model families, operations, state kinds, dtypes, topology, and execution mode.
- Add a pure Rust runtime core for capability admission, session transitions, requests, responses, and receipt construction.
- Add thin adapters for the current `llama-cpp-metalium` chat path and the persistent RWKV-7 command path.
- Add a deterministic fake adapter for positive and negative device-free tests.
- Add fail-closed source admission for future Rust Metalium bridges.
- Admit `Knight-Ops/libtt-metal-cxx` at commit `3f62391f9d1cb4d5bdf85266239f63a67f93b016` under repository-maintainer approval.
- Record the missing upstream SPDX identifier as metadata debt without blocking private repository use.
- Add BLAKE3-bound runtime receipts with explicit simulated, device-free, or physical execution classification.

## Impact

- **Rust**: a new pure runtime core and thin backend shells.
- **Configuration**: a typed Nickel runtime registry and checked-in canonical projection.
- **Commands**: existing chat and persistent commands gain explicit backend identities without changing their default behavior.
- **Evidence**: one common runtime receipt envelope.
- **Validation**: device-free contract, state-machine, adapter, source-admission, and receipt checks.

## Out of Scope

- A general tensor API or compiler IR.
- Burn, Candle, CubeCL, `mistral.rs`, or `zyx` model support.
- Claiming an exact SPDX license before upstream records one.
- Hardware performance claims or long simulator runs.
- A stable dynamic plugin ABI.
