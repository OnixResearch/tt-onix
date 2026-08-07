## Context

Upstream enables TT-Metal simulation with `TT_METAL_SIMULATOR`. The selected library must share a directory with `soc_descriptor.yaml`. Upstream also recommends slow dispatch and documents incomplete fast-dispatch determinism.

The repository currently installs all release libraries into one directory. It does not install the required descriptor or provide a TT-Metal wrapper. The package fetches the x86-style asset names only.

Multi-chip simulator support is experimental. Upstream requires a compatible TT-Metal branch and a cluster descriptor through `TT_METAL_MOCK_CLUSTER_DESC_PATH`.

## Decisions

### Decision: Package complete runtime directories

**Choice:** Create one immutable runtime directory per supported architecture and topology. Each directory contains one canonical `libttsim.so` link, the exact simulator asset, and the matching `soc_descriptor.yaml` from packaged TT-Metal.

**Rationale:** TT-Metal discovers the descriptor relative to the simulator library. A library-only package is not runnable.

### Decision: Select assets by host system

**Choice:** Use a typed Nix asset table keyed by host system, chip architecture, and topology. Fetch `_aarch64` assets only on verified aarch64 systems. Reject an unsupported combination during evaluation.

**Rationale:** One unsuffixed asset list cannot represent upstream's separate release binaries.

### Decision: Provide narrow wrappers

**Choice:** Add wrappers that set `TT_METAL_SIMULATOR` and `TT_METAL_SLOW_DISPATCH_MODE=1`. Set `TT_METAL_DISABLE_SFPLOADMACRO=1` only for paths that use the unsupported SFPU macro.

**Rationale:** Named wrappers make the tested environment explicit. Applications still use the same host and device binaries as silicon.

### Decision: Prove single-chip behavior first

**Choice:** Run one bounded Blackhole `1x1` smoke that opens one device, compiles firmware, executes representative TT-NN operations, and closes cleanly. Use a named timeout.

**Rationale:** A small smoke removes package and host-interface risks before expensive model simulation or hardware use.

### Decision: Keep multi-chip experimental

**Choice:** Expose `bh_x2` only after the package proves a matching cluster descriptor and TT-Metal interface. Do not treat a simulated P300 topology as physical P150x2 evidence.

**Rationale:** Upstream marks multi-chip support incomplete and branch-sensitive.

### Decision: Simulator receipts are explicit

**Choice:** Emit BLAKE3-bound receipts that name simulator version, library, descriptor, host system, TT-Metal identity, dispatch mode, workload, timeout, and outcome. Every receipt states that execution was simulated.

**Rationale:** Simulator evidence can prove software behavior only. It cannot prove silicon placement, performance, thermals, or health.

## Validation Strategy

Positive checks will cover typed asset selection, adjacent descriptor layout, exported ABI symbols, wrapper environment, and the Blackhole `1x1` smoke.

Negative checks will cover missing descriptors, wrong host assets, wrong chip descriptors, unsupported Quasar host paths, fast-dispatch claims, stale identities, and unapproved multi-chip configurations.

The smoke will use the package-matched TT-Metal source and a bounded representative path. It will not run full-model inference.

## Risks / Trade-offs

- Separate runtime directories duplicate small descriptor files.
- Simulator checks can take longer than normal device-free tests.
- Upstream release coverage can differ by host system and topology.
- Multi-chip support can remain blocked by the package-matched TT-Metal revision.
