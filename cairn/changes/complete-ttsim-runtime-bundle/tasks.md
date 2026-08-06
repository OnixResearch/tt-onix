# Tasks: Complete ttsim runtime bundle

## Phase 1: Authority and package layout

- [ ] [serial] B1 Record the current package build, asset table, TT-Metal identity, and Cairn validation before edits. r[tenstorrent.ttsim.runtime_bundle]
- [ ] [serial] A1 Build a typed host-system, architecture, topology, asset-name, and hash table from the pinned upstream release. r[tenstorrent.ttsim.asset_selection]
- [ ] [depends:tenstorrent.ttsim.asset_selection] A2 Add positive x86_64 and verified aarch64 selections plus negative unsupported-system and missing-asset fixtures. r[tenstorrent.ttsim.asset_selection]
- [ ] [depends:tenstorrent.ttsim.runtime_bundle] P1 Build one runtime directory per supported chip and topology with `libttsim.so` and adjacent `soc_descriptor.yaml`. r[tenstorrent.ttsim.runtime_bundle]
- [ ] [depends:tenstorrent.ttsim.runtime_bundle] P2 Add named wrappers for simulator path, slow dispatch, and the bounded SFPU workaround. r[tenstorrent.ttsim.runtime_bundle]

## Phase 2: Simulator validation

- [ ] [serial] V1 Add layout and ABI checks that reject a missing descriptor, wrong descriptor, missing symbol, or host-asset mismatch. r[tenstorrent.ttsim.runtime_bundle]
- [ ] [depends:tenstorrent.ttsim.runtime_bundle] S1 Add the bounded Blackhole `1x1` smoke with one open, firmware compilation, representative operations, and clean close. r[tenstorrent.ttsim.single_chip_smoke]
- [ ] [depends:tenstorrent.ttsim.single_chip_smoke] S2 Add positive and negative smoke fixtures for timeout, unsupported operation, nonzero exit, and incomplete close. r[tenstorrent.ttsim.single_chip_smoke]

## Phase 3: Experimental multi-chip boundary

- [ ] [depends:tenstorrent.ttsim.single_chip_smoke] M1 Audit package-matched TT-Metal for `bh_x2` support and record the exact supported or blocked interface. r[tenstorrent.ttsim.multi_chip_boundary]
- [ ] [depends:tenstorrent.ttsim.multi_chip_boundary] M2 If supported, package the exact cluster descriptor and add a bounded mesh-open and replication smoke. Otherwise, emit an exact blocker receipt. r[tenstorrent.ttsim.multi_chip_boundary]

## Phase 4: Evidence and documentation

- [ ] [parallel] E1 Add simulated receipts that bind library, descriptor, host, TT-Metal, dispatch, workload, timeout, and outcome identities. r[tenstorrent.ttsim.evidence]
- [ ] [parallel] E2 Add negative receipt fixtures for a hardware claim, stale identity, missing non-claim, and unsupported topology. r[tenstorrent.ttsim.evidence]
- [ ] [serial] D1 Document supported systems, wrappers, timeout policy, multi-chip status, simulator workarounds, and non-claims. r[tenstorrent.ttsim.evidence]
- [ ] [depends:tenstorrent.ttsim.single_chip_smoke] G1 Run formatting, package checks, the bounded smoke, Cairn validation, and all change gates. r[tenstorrent.ttsim.single_chip_smoke]

## Verification Coverage

- `Scenario: Supported host asset is selected` -> A1, A2
- `Scenario: Unsupported host asset is rejected` -> A1, A2
- `Scenario: Complete simulator runtime evaluates` -> P1, P2, V1
- `Scenario: Incomplete simulator runtime fails` -> P1, V1
- `Scenario: Blackhole single-chip smoke passes` -> S1, S2
- `Scenario: Simulator smoke failure stays bounded` -> S2, E1
- `Scenario: Multi-chip path needs matched authority` -> M1, M2
- `Scenario: Simulator evidence preserves non-claims` -> E1, E2
