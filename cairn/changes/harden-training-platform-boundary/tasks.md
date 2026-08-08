# Tasks: Harden training platform boundary

## Phase 1: Baseline and authority

- [x] [serial] B1 Record the current focused checks, the two admission-bypass fixtures, the driver mismatch, and Cairn validation before edits. r[tenstorrent.training_platform.runtime_bundle]
- [x] [serial] A1 Export a closed, versioned admission policy from Nickel and consume it in the pure Rust core. r[tenstorrent.training_platform.configuration]
- [x] [depends:tenstorrent.training_platform.configuration] A2 Reject wrong schema versions, nested unknown fields, wrong package names, wrong ABIs, invalid dates, unsafe paths, unsupported board classes, zero fabric channels, mutable firmware values, and incomplete evidence exclusions. r[tenstorrent.training_platform.configuration]
- [x] [depends:tenstorrent.training_platform.runtime_bundle] A3 Bind production launch to the exact package-owned manifest, configuration, descriptor, and wheel-set identities. r[tenstorrent.training_platform.runtime_bundle] r[tenstorrent.training_platform.launch]

## Phase 2: Host and preflight

- [x] [serial] H1 Package the admitted TT-KMD version or revise the manifest only after package-matched compatibility evidence. r[tenstorrent.training_platform.host_module]
- [x] [depends:tenstorrent.training_platform.host_module] H2 Add the configurable driver package, driver assertion, descriptor existence assertion, named hugepage setup, public example export, and group-restricted access mode. r[tenstorrent.training_platform.host_module]
- [x] [depends:tenstorrent.training_platform.preflight] P1 Move preflight into the production launcher and reject duplicate device identities, busy devices, wrong firmware, wrong mesh, wrong fabric policy, or missing hugepages before Docker starts. r[tenstorrent.training_platform.preflight]
- [x] [depends:tenstorrent.training_platform.preflight] P2 Bind the same-process preflight result to the launch plan and terminal receipt. r[tenstorrent.training_platform.launch]

## Phase 3: Hermetic runtime and evidence

- [x] [depends:tenstorrent.training_platform.runtime_bundle] R1 Fetch the complete Python wheel closure with exact hashes and remove runtime package-index access and Pip upgrades. r[tenstorrent.training_platform.runtime_bundle]
- [x] [depends:tenstorrent.training_platform.runtime_bundle] R2 Install only from the read-only wheelhouse with hash checking in platform, single-card LoRA, and dual-card LoRA containers. r[tenstorrent.training_platform.launch]
- [x] [serial] E1 Separate private logs from portable evidence and generate sanitized aggregate artifacts before hashing. r[tenstorrent.training_platform.evidence]
- [x] [depends:tenstorrent.training_platform.evidence] E2 Add the terminal stage-state core and one exit trap that emits validated success and failure receipts. r[tenstorrent.training_platform.evidence] r[tenstorrent.training_platform.launch]
- [x] [depends:tenstorrent.training_platform.evidence] E3 Replace checked-in platform evidence with a sanitized receipt and record existing board identifiers as previously disclosed. r[tenstorrent.training_platform.evidence]

## Phase 4: Verification and documentation

- [x] [parallel] V1 Add positive fixtures for exact authority, complete policy validation, offline installation, same-process preflight, sanitized evidence, and terminal receipts. r[tenstorrent.training_platform.verification]
- [x] [parallel] V2 Add negative fixtures for arbitrary manifests, invalid nested configuration, network access, duplicate devices, private artifact content, missing stages, and stale receipts. r[tenstorrent.training_platform.verification]
- [x] [serial] D1 Update operator documentation with the exact host profile, offline closure, private-log location, failure receipt rules, and non-claims. r[tenstorrent.training_platform.verification]
- [x] [depends:tenstorrent.training_platform.verification] V3 Run Rust tests, ShellCheck, NixOS evaluation, focused Nix checks, Cairn validation, and all change gates. r[tenstorrent.training_platform.verification]
- [x] [depends:tenstorrent.training_platform.verification] V4 Run one bounded dual-P150A smoke and admit the sanitized replacement receipt. r[tenstorrent.training_platform.verification]

V4 passed on 2026-08-08 with TT-KMD `2.10.0` and firmware `18.8.0.0`.
The preflight admitted two P150A devices and the `p150_x2` mesh. All seven
receipt stages passed. The checked-in replacement contains only sanitized
portable artifacts. A failed preliminary run exposed a `fuser` visibility
gap for root-owned processes. The final launcher also checks the KMD procfs
PID files and fails closed before Docker when an owner exists.

## Verification Coverage

- `Scenario: Reviewed runtime authority is required` -> A1, A2, A3, V2
- `Scenario: Runtime dependency closure is offline` -> R1, R2, V1
- `Scenario: Invalid nested configuration fails admission` -> A1, A2, V2
- `Scenario: Preflight and launch remain one operation` -> P1, P2, V1
- `Scenario: Incompatible host profile fails evaluation` -> H1, H2, V2
- `Scenario: Portable evidence excludes private content` -> E1, E3, V2
- `Scenario: Every exit has a terminal receipt` -> E2, V1, V2
