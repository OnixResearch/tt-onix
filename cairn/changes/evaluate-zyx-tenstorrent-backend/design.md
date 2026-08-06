## Context

At commit `7606188adac64b362c257159af9949d860759cc0`, `zyx` has a `tenstorrent` Cargo feature and three main implementation surfaces:

- `zyx/src/backend/tenstorrent.rs`
- `zyx/src/codegen/tenstorrent.rs`
- `zyx/src/kernel/tenstorrent.rs`

The build script compiles `tt_runtime.cpp` with `g++`. It expects `TT_METAL_ROOT`, `build_Release`, and a `.cpmcache` spdlog tree. It writes `zyx-tt-runtime` under `XDG_CONFIG_HOME` or `$HOME/.config`.

The backend starts a child runtime and exchanges JSON messages. Its cleanup path can run `pkill -9 zyx-tt-runtime`. Its capability declaration uses all dtypes although reviewed code generation supports a smaller set. Upstream marks maintenance as experimental and says that the full test suite does not pass.

## Decisions

### Decision: Evaluate one exact upstream revision

**Choice:** Pin version `0.16.0` at commit `7606188adac64b362c257159af9949d860759cc0`. Bind the source URL, revision, Nix source hash, BLAKE3 tree identity, Cargo lock identity, and LGPL-3.0-only license.

**Rationale:** The backend changes quickly. A branch name cannot support a reproducible result.

### Decision: Keep a small auditable patch queue

**Choice:** Store separate patches for Nix build paths, runtime discovery, process ownership, and truthful capabilities. Each patch names the upstream file and reason.

Do not copy backend code into a new local implementation.

**Rationale:** Separate patches expose divergence and simplify later upstream refreshes.

### Decision: Build and install the runtime through Nix

**Choice:** Build `zyx-tt-runtime` in Cargo `OUT_DIR`. Install it under `$out/libexec/zyx`. Pass packaged TT-Metal include, library, and runtime roots through explicit build variables.

Runtime lookup uses a package wrapper or one explicit runtime path. The build and tests cannot write to `$HOME`, `XDG_CONFIG_HOME`, or `/tmp` except through declared test directories.

**Rationale:** Nix owns every compiler and dependency. A Cargo build must not mutate a user configuration directory.

### Decision: Enforce exact child ownership

**Choice:** Keep the runtime child handle in the imperative backend shell. On close, error, or timeout, terminate and reap that exact child. Remove all `pkill`, process-name search, and global signal behavior.

The pure protocol core validates messages and lifecycle transitions without process access.

**Rationale:** Global termination can kill another user or test session. Exact ownership also makes parallel tests safe.

### Decision: Publish a reviewed capability matrix

**Choice:** Declare each dtype and operation as `supported`, `unsupported`, or `unknown`. Bind every supported entry to one device-free code-generation fixture. Simulator evidence can promote an entry to `simulated`, but not to physical support.

The initial matrix covers allocation, transfer, unary, binary, movement, reduction, matmul, and realization paths. Unknown and incomplete paths remain fail-closed.

**Rationale:** Backend availability does not prove operation coverage.

### Decision: Separate compiler checks from simulator checks

**Choice:** Default flake checks compile the package and run bounded CPU, graph, lowering, and Tenstorrent code-generation fixtures. They do not load a simulator or device.

A separate check runs one Blackhole `1x1` ttsim smoke with slow dispatch and a named timeout. It covers runtime startup, one transfer, one supported operation, result readback, and clean close.

**Rationale:** Compiler checks are fast and deterministic. Simulator startup can be slow and remains outside the default device-free gate.

### Decision: Produce an evaluation verdict, not a support flag

**Choice:** The receipt reports `blocked` or `eligible_for_runtime_adapter_spike`. Eligibility requires hermetic packaging, safe cleanup, exact capabilities, passing device-free fixtures, and a passing simulator smoke.

The receipt binds simulator classification and exact gaps. It never exposes a runnable model target.

**Rationale:** The evaluation can succeed by proving that the backend is not ready. This prevents pressure to convert partial evidence into a support claim.

## Validation Strategy

Positive device-free tests cover source admission, lockfile use, package build, runtime path resolution, process lifecycle transitions, and supported code-generation fixtures.

Negative tests cover source drift, an unlocked dependency graph, user-directory writes, checkout path assumptions, global process termination, unsupported dtypes, unsupported operations, malformed JSON, duplicate close, timeout, and false physical claims.

The optional simulator smoke uses the package-matched TT-Metal and ttsim runtime. It records every version and path identity. It does not run model inference.

## Risks / Trade-offs

- The package-matched TT-Metal API can differ from the upstream revision used by `zyx`. The verdict records the exact compile or runtime mismatch.
- LGPL obligations apply to distributed binaries and modified source. Package metadata and documentation preserve the upstream license and patch sources.
- A small operation can pass while model execution remains blocked. The verdict grants only adapter-spike eligibility.
- Simulator behavior cannot prove silicon placement or performance.

## Non-Goals

- Make the entire `zyx` test suite pass.
- Add a complete LLM, RWKV-7, or Qwen implementation.
- Claim training, inference, quantization, or multi-device support.
- Change TT-Metal, firmware, driver, or simulator pins to fit `zyx`.
