## Context

The package-matched runtime uses TT-Metal 0.74.0 and a P150x2 mesh descriptor. The stale runtime replicates checkpoint tensors across physical devices 0 and 1. It executes embedding, twelve layers, recurrent state updates, final normalization, the untied head, and greedy selection through TT-NN operations.

Attempt 3 established owner isolation and device discovery. Its compile command contained both `-DPROFILE_KERNEL=1` and `-DWATCHER_ENABLED`. The package-matched TT-Metal source enables Watcher whenever `TT_METAL_WATCHER` is present, including the value `0`. The profiler documentation requires Watcher and DPrint to be unset during profiling.

## Decisions

### Decision: Port reusable packages instead of the stale branch

**Choice:** Copy the full-device runtime and evidence classifier into `tenstorrent.nix`. Adapt their package wiring to the current flake. Do not merge the stale `onix-core` branch or its machine-specific runbooks.

**Rationale:** The standalone repository owns reusable package behavior. The old branch predates that authority migration and contains unrelated machine state.

### Decision: Model profiler environment changes as a pure policy

**Choice:** Build a deterministic list of environment mutations in a pure Rust core. A thin command shell applies those mutations before `exec`. The policy sets Inspector and profiler values and removes Watcher and DPrint variables.

**Rationale:** Unit tests can inspect the exact process policy without executing the device runtime. This boundary prevents disabled-looking values from silently enabling presence-based TT-Metal options.

### Decision: Reject every present Watcher or DPrint value

**Choice:** The C++ runtime captures an environment snapshot, then passes it to a pure validator. The validator rejects `TT_METAL_WATCHER` and `TT_METAL_DPRINT_CORES` whenever either variable is present. Values such as `0` and an empty string are not exceptions.

**Rationale:** The package-matched documentation says these variables must not be set simultaneously with profiling. TT-Metal parses Watcher by presence.

### Decision: Retain a bounded attempt-3 classifier fixture

**Choice:** Keep the decisive attempt-3 output, safety receipt, terminal outcome, and board records in a deterministic fixture. Validate that both devices opened, both profiler and Watcher macros appeared, BRISC overflowed, the process failed once, and no success is claimed.

**Rationale:** The fixture records the reason for this change without importing large reproducible CPU-reference artifacts or changing the raw terminal root.

### Decision: Defer hardware execution

**Choice:** End this change after device-free validation and repository integration. Create a fresh machine-bound change before owner isolation or device execution.

**Rationale:** A new attempt must bind the current system, current owner set, a new evidence root, and a reviewed restoration policy.

## Risks / Trade-offs

- The replicated runtime proves execution on both cards but does not improve single-request performance.
- Static source and closure checks cannot prove physical placement. A later hardware receipt must provide that evidence.
- The stale C++ runtime can expose new compile or API drift during the current package build. Device-free builds must fail closed.
- The retained fixture proves only the recorded attempt-3 failure boundary.

## Validation Notes

The focused runtime, classifier, source, closure, retained attempt-3, framework-parity, model-carry, and migration checks pass. The evidence crate also passes Clippy with warnings denied.

The full flake check evaluates all outputs but stops at the existing `rwkv-ttwkv7-boundary-device` check. That check also fails unchanged on `origin/main`. Its fixed closure-count contract is outside this change. The layer harness also has two pre-existing `too_many_arguments` Clippy failures in `src/observed_layer.rs`; this change does not modify that file.
