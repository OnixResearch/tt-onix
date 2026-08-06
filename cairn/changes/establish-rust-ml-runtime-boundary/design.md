## Context

The repository already separates much RWKV-7 logic into pure Rust cores. `pkgs/rwkv-layer-harness/src/dispatch_abi.rs` validates bounded frames and persistent session transitions. Thin binaries own child processes, sockets, files, and device access.

`pkgs/rwkv7-chat/default.nix` has a different boundary. It invokes `llama-cli` directly and embeds Metalium-specific environment rules. The full-model P150x2 path uses `rwkv7-p150x2-command` and a separate evidence schema.

The reviewed `Knight-Ops/libtt-metal-cxx` tree exposes useful TT-Metal host APIs through `cxx`. Commit `3f62391f9d1cb4d5bdf85266239f63a67f93b016` has no license file or Cargo license field.

The repository maintainer confirms that the bridge is open-source Tenstorrent work and approves its use here. This approval permits repository integration. It does not identify an exact SPDX license.

## Decisions

### Decision: Define a model-runtime boundary, not a tensor framework

**Choice:** The core exposes model load, prefill, decode, reset, close, capability query, and evidence operations. Runtime state stays behind an opaque session identity.

**Rationale:** These operations cover current chat and persistent decode paths. Tensor allocation and kernel compilation belong to backend implementations.

### Decision: Keep policy in Nickel and transitions in pure Rust

**Choice:** Nickel defines runtime identities, model families, supported operations, state kinds, dtypes, topologies, and execution modes. A checked-in canonical projection feeds a pure Rust core.

The core accepts values and returns decisions. It does not read files, environment variables, sockets, clocks, processes, or devices.

**Rationale:** Nickel rejects invalid declarations during evaluation. Pure Rust state transitions support deterministic tests without hardware or mocks.

### Decision: Start with two current adapters and one fake adapter

**Choice:** Add thin adapters for `llama-cpp-metalium` and `rwkv7-p150x2-command`. Add a deterministic fake adapter that implements the same boundary in memory.

The adapters own process launch, environment setup, timeout handling, and cleanup. Each adapter can terminate only the child that it created.

**Rationale:** Current adapters prove that the contract fits two different execution paths. The fake adapter covers success and failure without a device.

### Decision: Keep the dispatch ABI behind the adapter

**Choice:** Reuse the current bounded frame parser and persistent lifecycle where it applies. Do not make its RWKV-specific frame format the repository-wide runtime protocol.

**Rationale:** The existing ABI has strong validation, but its layer and state fields are model-specific. The common boundary must also admit KV-cache and hybrid-state runtimes.

### Decision: Bind source authorization and license metadata separately

**Choice:** Add a candidate-source manifest with URL, revision, content identity, authorization basis, license status, API scope, and admission result.

An explicit upstream license or repository-maintainer approval can authorize use. The initial manifest admits `libtt-metal-cxx` under maintainer approval. It records the missing SPDX identifier as metadata debt.

The package can fetch, patch, build, test, and expose the approved source. It cannot claim a specific SPDX license until upstream records one.

**Rationale:** Local authority permits this private repository to use the source. Separate metadata prevents the package from inventing legal terms.

### Decision: Make capability claims exact

**Choice:** Each adapter declares only tested model families, state kinds, dtypes, operations, topologies, and execution modes. Runtime admission rejects a request outside this declaration before process launch.

**Rationale:** A generic available flag hides partial support. Exact capabilities prevent a backend experiment from becoming a support claim.

### Decision: Use one common runtime receipt

**Choice:** The receipt binds the runtime profile, adapter revision, model identity, requested operations, state kind, topology, execution classification, result, and exact blocker. BLAKE3 binds canonical receipt bytes.

A physical classification requires existing physical evidence. A simulator result always states that it is simulated.

**Rationale:** Consumers need one comparison surface without losing backend-specific evidence.

## Validation Strategy

Positive device-free tests cover valid Nickel export, fake session transitions, current adapter command construction, capability admission, and deterministic receipts.

Negative tests cover unknown runtimes, unsupported operations, invalid transitions, duplicate close, timeout, malformed adapter output, global process termination, missing source authority, and false physical claims.

Focused checks verify that the pure core has no process, environment, filesystem, socket, clock, or device access. Existing runtime and chat defaults must keep their current behavior.

## Risks / Trade-offs

- The first contract can omit operations needed by a later runtime. Versioned capability data permits additive updates.
- A common boundary can hide backend details. Backend-specific evidence remains available through typed extension fields.
- Process adapters add one translation layer. The pure core keeps this layer small and deterministic.
- The exact `libtt-metal-cxx` SPDX identifier remains unknown. Package metadata records that fact without blocking approved use.

## Non-Goals

- Replace TT-NN, TT-Metal, llama.cpp, or the current dispatch ABI.
- Standardize tensor memory layouts or kernel languages.
- Add runtime hot loading or an unstable Rust trait-object ABI.
- Run hardware or long simulator workloads in the default gate.
