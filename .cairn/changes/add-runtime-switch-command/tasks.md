# Tasks: `tt-model switch` and the reviewable host fragment

## Phase 1: Contract

- [x] [serial] C1 Add an optional `containment` block (repository-relative command, arguments expanding only `@package@` and `@package_identity@`, verdict field and value) and an optional `readiness.identity_source`/`request_path`. r[tenstorrent.runtime_profile.switch.containment]
- [x] [serial] C2 Record qwen38's containment checker and its decode-evidence identity source in the registry, add the placeholder negative fixture, and extend the registry check to prove the containment command exists. r[tenstorrent.runtime_profile.switch.containment]

## Phase 2: Switch

- [x] [serial] S1 Refactor the record writer into `build_record` and `commit_record`, and add the mesh and verified-identity fields. r[tenstorrent.runtime_profile.switch.valid]
- [x] [serial] S2 Implement the sequence: lock, check, cards, stop, reset, re-check, containment, candidate record, start, readiness, identity, verdict. r[tenstorrent.runtime_profile.switch]
- [x] [serial] S3 Implement readiness polling and identity readback from a declared source, comparing model identity, package identity, device ids, `simulated`, runtime identity, and profile id. r[tenstorrent.runtime_profile.switch.identity] r[tenstorrent.runtime_profile.switch.readiness]
- [x] [serial] S4 Refuse a job profile, a failed or structural check, a foreign holder, a missing or failing helper, a failed start, a held lock, and an unattended force; keep containment and structural failures unbypassable. r[tenstorrent.runtime_profile.switch.mode] r[tenstorrent.runtime_profile.switch.check] r[tenstorrent.runtime_profile.switch.holder] r[tenstorrent.runtime_profile.switch.helper] r[tenstorrent.runtime_profile.switch.force] r[tenstorrent.runtime_profile.switch.unattended] r[tenstorrent.runtime_profile.switch.lock]
- [x] [serial] S5 Implement rollback: restore previous bytes, restart and verify the previous owner, or withdraw the record and keep it in history. r[tenstorrent.runtime_profile.switch.rollback] r[tenstorrent.runtime_profile.switch.rollback.valid] r[tenstorrent.runtime_profile.switch.rollback.failed]

## Phase 3: Host fragment

- [x] [serial] H1 Add `nixos/examples/qwen38-tt-model-switch.nix` with the reviewed serving options, the competing owners, and the helper grant. r[tenstorrent.runtime_profile.switch.host]
- [x] [serial] H2 Add `checks/tt-model-host-config.nix`: one added unit, one sudo rule, the serving options, the unchanged competing owners, and the wide-grant negative. r[tenstorrent.runtime_profile.switch.host.valid] r[tenstorrent.runtime_profile.switch.host.wide]

## Phase 4: Verification

- [x] [serial] V1 Run the `tt-model` install check with twelve switch scenarios, including byte-identical rollback and the containment refusal. r[tenstorrent.runtime_profile.switch.valid] r[tenstorrent.runtime_profile.switch.rollback.valid]  (evidence/switch-scenarios.md)
- [x] [serial] V2 Build the host-configuration and sudo checks. r[tenstorrent.runtime_profile.switch.host.valid]  (evidence/host-fragment.md)
- [x] [serial] V3 Build the registry check with the containment fixture. r[tenstorrent.runtime_profile.switch.containment]  (evidence/switch-scenarios.md)
- [x] [serial] V4 Regenerate the manifest and run the aggregate quality gate. r[tenstorrent.runtime_profile.switch.host]  (evidence/quality.md)

## Verification Coverage

- `Scenario: A healthy switch commits a verified record` -> S1, S2, S3, V1
- `Scenario: The profile is a job` -> S4, V1
- `Scenario: The check fails` -> S4, V1
- `Scenario: A card is held outside the owner` -> S4, V1
- `Scenario: The helper is absent, fails, or the start fails` -> S4, V1
- `Scenario: Readiness does not arrive` -> S3, V1
- `Scenario: The instance reports another identity` -> S3, S5, V1
- `Scenario: Containment fails` -> C1, C2, S2, V1
- `Scenario: Another switch holds the lock` -> S4, V1
- `Scenario: A forced switch is unattended` -> S4, V1
- `Scenario: Only admission failed and the caller forced it` -> S4, V1
- `Scenario: A failed switch leaves the previous record unchanged` -> S5, V1
- `Scenario: The previous owner cannot be restored` -> S5, V1
- `Scenario: The fragment renders the intended host` -> H1, H2, V2
- `Scenario: A wider grant is configured` -> H2, V2
