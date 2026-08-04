# Tasks: Dual-P150A TT-XLA training platform

## Phase 1: Runtime authority

- [ ] [serial] P1 Record the current flake, driver, firmware, device, and mesh baseline without mutation. r[tenstorrent.training_platform.runtime_bundle] r[tenstorrent.training_platform.preflight]
- [ ] [serial] P2 Select one compatible TT-XLA training bundle and record every immutable source, package, image, ABI, driver, and firmware identity. r[tenstorrent.training_platform.runtime_bundle]
- [ ] [depends:tenstorrent.training_platform.runtime_bundle] P3 Package the exact runtime image, TT-Blacksmith source, compatibility manifest, and Nix-owned launcher. r[tenstorrent.training_platform.runtime_bundle] r[tenstorrent.training_platform.launch]

## Phase 2: Configuration and preflight

- [ ] [parallel] P4 Add typed Nickel contracts for runtime, device, mesh, hugepage, cache, output, and evidence configuration. r[tenstorrent.training_platform.configuration]
- [ ] [depends:tenstorrent.training_platform.configuration] P5 Export deterministic runtime data and reject unknown fields, mutable identities, invalid paths, and unsupported combinations. r[tenstorrent.training_platform.configuration] r[tenstorrent.training_platform.verification]
- [ ] [parallel] P6 Add pure manifest, topology, launch-plan, and receipt admission functions with plain unit tests. r[tenstorrent.training_platform.runtime_bundle] r[tenstorrent.training_platform.preflight] r[tenstorrent.training_platform.evidence]
- [ ] [depends:tenstorrent.training_platform.preflight] P7 Add the thin hardware shell for device discovery, ownership, hugepages, mesh open, smoke execution, health capture, mesh close, and restoration. r[tenstorrent.training_platform.preflight] r[tenstorrent.training_platform.launch]
- [ ] [depends:tenstorrent.training_platform.launch] P8 Add the NixOS dual-P150A training example with `meshName = "p150_x2"` and named hugepage requirements. r[tenstorrent.training_platform.configuration]

## Phase 3: Validation and evidence

- [ ] [parallel] V1 Add positive fixtures for the pinned bundle, valid configuration, dual-card topology, launch plan, clean smoke run, and restoration. r[tenstorrent.training_platform.verification]
- [ ] [parallel] V2 Add negative fixtures for mutable tags, stale identities, ABI mismatch, invalid device counts, wrong architecture, old firmware, missing hugepages, wrong mesh, missing links, busy devices, and failed restoration. r[tenstorrent.training_platform.verification]
- [ ] [depends:tenstorrent.training_platform.launch] V3 Run the bounded hardware smoke on two connected P150A cards and save the BLAKE3-bound receipt. r[tenstorrent.training_platform.evidence] r[tenstorrent.training_platform.verification]
- [ ] [serial] P9 Document runtime selection, hardware prerequisites, cable and descriptor requirements, recovery, exact commands, and non-claims. r[tenstorrent.training_platform.documentation]
- [ ] [depends:tenstorrent.training_platform.verification] V4 Run formatting, focused tests, Nix package builds, NixOS evaluation, Cairn validation, and all lifecycle gates. r[tenstorrent.training_platform.verification]

## Verification Coverage

- `Scenario: Immutable compatible bundle is admitted` -> P2, P3, V1
- `Scenario: Mutable or incompatible bundle is rejected` -> P2, P6, V2
- `Scenario: Valid training configuration exports deterministically` -> P4, P5, V1
- `Scenario: Invalid training configuration fails admission` -> P5, V2
- `Scenario: Dual-P150A preflight passes` -> P7, V3
- `Scenario: Unsafe hardware state blocks launch` -> P7, V2
- `Scenario: Launcher preserves runtime isolation` -> P3, V1, V2
- `Scenario: Hardware receipt binds bounded evidence` -> P6, V3
- `Scenario: Documentation preserves the platform boundary` -> P9, V4
