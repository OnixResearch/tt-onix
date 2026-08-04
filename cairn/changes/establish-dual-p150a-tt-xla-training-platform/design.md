# Design: Dual-P150A TT-XLA training platform

## Context

The repository has a package-matched TT-Metal runtime and a NixOS device module. The module already exports `TT_MESH_GRAPH_DESC_PATH` from `hardware.tenstorrent.meshName`.

TT-Metal `v0.74.0-rc6` ships `p150_x2_mesh_graph_descriptor.textproto`. The descriptor defines a Blackhole `1x2` device mesh with four fabric channels.

Each P150A has 32 GB of GDDR6 and four passive QSFP-DD ports. Public Tenstorrent documentation permits single-card and multi-card P150A operation.

TT-Blacksmith uses TT-XLA for PyTorch training. Its upstream quick start requires all Tenstorrent devices inside the container.

The TT-XLA runtime has its own compatibility set. This change must not mix that set with the existing TT-Metal closure without evidence.

## Decisions

### Decision: Use TT-XLA and TT-Blacksmith as the first training route

**Choice:** Use PyTorch/XLA through TT-XLA and adapt TT-Blacksmith experiment structure for model work.

**Rationale:** Upstream TT-Blacksmith already demonstrates LoRA training on P150 and Blackhole meshes. Native TT-NN training requires more framework and optimizer work before RWKV bring-up.

### Decision: Pin one complete compatibility bundle

**Choice:** Record immutable identities for every runtime component in one manifest. Use source revisions, image digests, package hashes, driver versions, and firmware requirements.

**Rationale:** Independent latest versions can have incompatible ABIs. The manifest gives one reviewable runtime identity and one failure boundary.

The current TT-Metal package remains an authority for current inference packages. It does not become TT-XLA authority without compatibility evidence.

### Decision: Use a pinned upstream image as the bootstrap boundary

**Choice:** Fetch one upstream TT-XLA image by digest. Fetch TT-Blacksmith by revision. Expose both through a Nix-owned launcher.

**Rationale:** Upstream documents this runtime path. An exact image digest removes the mutable `latest` reference while native packaging remains a later optimization.

The launcher passes all Tenstorrent devices and the required hugepage mount. It rejects ambient Python and TT-Metal paths that can alter the selected bundle.

### Decision: Author runtime configuration in typed Nickel

**Choice:** Define contracts for the runtime bundle, device count, architecture, mesh shape, mesh descriptor, hugepage mount, cache roots, output roots, and evidence policy.

**Rationale:** Nickel can reject invalid combinations before launch. Generated data uses JSON or environment variables only where upstream tools require them.

### Decision: Require a fail-closed dual-P150A preflight

**Choice:** Validate these facts before a dual-card launch:

- Exactly two selected Blackhole P150-class devices are available.
- Each selected device reports supported firmware and driver compatibility.
- `TT_MESH_GRAPH_DESC_PATH` names the packaged `p150_x2` descriptor.
- The runtime opens a `1x2` mesh and closes it cleanly.
- The fabric exposes the configured P150-X2 channel policy.
- No other process owns either selected device.

**Rationale:** A compiler result cannot establish physical topology or safe device ownership. A separate preflight gives a clear blocker before model compilation.

### Decision: Separate pure admission from device access

**Choice:** Put manifest, configuration, topology, and receipt admission in pure deterministic functions. Keep device discovery, mounts, process launch, and file writes in thin shells.

**Rationale:** Device-free fixtures can validate policy without hardware. Hardware shells remain small and auditable.

### Decision: Emit bounded BLAKE3 receipts

**Choice:** Emit one launch-plan receipt and one hardware receipt. Bind configuration, runtime manifest, mesh descriptor, selected devices, command identity, status, and artifact identities with BLAKE3.

**Rationale:** Later Cairns need exact platform evidence. Raw logs, serial numbers, credentials, and absolute private paths stay outside portable receipts.

## Test Design

Positive fixtures cover an admitted compatibility bundle, valid Nickel configuration, two P150A devices, a `1x2` mesh, required fabric links, and clean ownership restoration.

Negative fixtures cover mutable image tags, stale revisions, ABI mismatches, one device, extra selected devices, wrong architecture, old firmware, missing hugepages, wrong descriptors, missing links, busy devices, launch failure, and close failure.

The hardware smoke opens the mesh, runs a bounded collective-free tensor operation, closes the mesh, and records device health before and after.

## Risks and Trade-offs

- The pinned image is not a native Nix build. The receipt states this boundary.
- TT-XLA can require a newer TT-Metal or driver set than the current packages. The compatibility manifest must resolve this conflict explicitly.
- TT-Blacksmith cannot isolate individual devices inside its container. The launcher passes all devices and applies selection inside the supported runtime API.
- A relaxed mesh descriptor can open with fewer links. The training profile can require a stricter observed link policy.

## Claim Boundary

A passing platform receipt proves the selected runtime identity and the observed dual-card preflight. It does not prove training correctness, gradient parity, convergence, speedup, thermal endurance, or future compatibility.

## Authority Sources

- `https://docs.tenstorrent.com/tt-blacksmith/`
- `https://docs.tenstorrent.com/tt-xla/`
- `.pi/references/tenstorrent/docs-site/core/aibs/blackhole/index.md`
- `.pi/references/tenstorrent/tt-metal-docs/tt_metal/fabric/mesh_graph_descriptors/p150_x2_mesh_graph_descriptor.textproto`
- `.pi/references/tenstorrent/tt-metal-docs/tt_metal/api/tt-metalium/cluster.hpp`
