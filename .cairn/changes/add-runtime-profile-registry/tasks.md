# Tasks: Runtime-profile registry and read-only status

## Phase 1: Registry

- [x] [serial] R1 Transcribe the four runtime profiles from their sources: the Qwen environment and argv from `nixos/module.nix` and the served drop-in, the nanbeige runner's required arguments and cache keying, the rwkv7 TT-NN wrapper's mesh descriptor, and the native gate wrappers. Cite the source on every block. r[tenstorrent.runtime_profile.registry]
- [x] [serial] R2 Add the closed typed contract with field and cross-field rules, each with a message a fixture can assert. r[tenstorrent.runtime_profile.contract]
- [x] [serial] R3 Export the canonical JSON under the registered generated root and commit it. r[tenstorrent.runtime_profile.registry.fresh]
- [x] [serial] R4 Add the freshness check: regenerate and compare, recompute every authority and receipt digest, and cross-check the model-target identity pins. r[tenstorrent.runtime_profile.registry.fresh.valid]
- [x] [serial] R5 Add seven negative fixtures: unknown field, missing field, wrong mode, malformed digest, serve ids not `[0,1]`, serve without a port, duplicate name. r[tenstorrent.runtime_profile.contract.closed] r[tenstorrent.runtime_profile.contract.required] r[tenstorrent.runtime_profile.contract.mode] r[tenstorrent.runtime_profile.contract.hash] r[tenstorrent.runtime_profile.contract.serve_mesh] r[tenstorrent.runtime_profile.contract.serve_port] r[tenstorrent.runtime_profile.registry.unique]
- [x] [serial] R6 Register the check in the closed required-check catalog and regenerate the catalog export. r[tenstorrent.runtime_profile.registry.fresh]

## Phase 2: Status

- [x] [serial] S1 Add `pkgs/tt-model` with `status [--json]`: no root, no sudo, no device open, no write, no unit mutation. r[tenstorrent.runtime_profile.status]
- [x] [serial] S2 Read the root-owned record and prove it is rooted: package exists, store name matches the profile, GC root resolves to it. r[tenstorrent.runtime_profile.status.managed]
- [x] [serial] S3 Read the legacy sources: the operator symlink, the Nix profile generation and the packages its manifest installs, and the unit state; report `unavailable` instead of failing when a source cannot be read. r[tenstorrent.runtime_profile.status.unmanaged]
- [x] [serial] S4 Report dangling, unrooted, and contradictory sources with their path and reason, and exit non-zero. r[tenstorrent.runtime_profile.status.dangling]
- [x] [serial] S5 Refuse an unknown profile or a registry that does not match its contract with a usage error. r[tenstorrent.runtime_profile.status.invalid]
- [x] [serial] S6 Assert in the package check that the program text never names a device node, `tt-smi`, or the owner helper, and that the shipped registry is the reviewed export. r[tenstorrent.runtime_profile.status]

## Phase 3: Verification

- [x] [serial] V1 Build the registry check and record its summary and each fixture refusal. r[tenstorrent.runtime_profile.registry.fresh.valid]  (evidence/registry-check.md)
- [x] [serial] V2 Build `tt-model` and run its install check, which exercises every verdict and refusal. r[tenstorrent.runtime_profile.status.managed] r[tenstorrent.runtime_profile.status.dangling]  (evidence/package-check.md)
- [x] [serial] V3 Run `tt-model status` and `status --json` on the host and record the real legacy facts. r[tenstorrent.runtime_profile.status.dangling]  (evidence/status-host.md)
- [x] [serial] V4 Prove the model-target registry and the qwen38 derivation are unchanged by digest and derivation path. r[tenstorrent.runtime_profile.registry.valid]  (evidence/identity-invariance.md)
- [x] [serial] V5 Run the aggregate quality gate, including Cairn validation and the closed required-check catalog. r[tenstorrent.runtime_profile.registry.fresh]

## Verification Coverage

- `Scenario: Every switchable runtime has exactly one profile` -> R1, R3, V4
- `Scenario: A profile name is duplicated` -> R5
- `Scenario: An undocumented field is added` -> R5, V1
- `Scenario: A required field is removed` -> R5, V1
- `Scenario: A closed set is extended by hand` -> R5, V1
- `Scenario: A digest is not lowercase hexadecimal` -> R5, V1
- `Scenario: A serve profile does not own the admitted mesh` -> R5, V1
- `Scenario: A serve profile has no port` -> R5, V1
- `Scenario: The registry is fresh and every reference resolves` -> R3, R4, V1
- `Scenario: The export or a reference drifts` -> R4, R5, V1
- `Scenario: A rooted record is the active runtime` -> S2, S4, V2
- `Scenario: A recorded package or a legacy pointer is gone` -> S4, V1, V2, V3
- `Scenario: Nothing claims the cards` -> S3, V2
- `Scenario: The registry does not match its contract` -> S5, V2
