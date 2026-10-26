# Tasks: `run`, the privileged-helper boundary, and the staged sudo grant

## Phase 1: Boundary

- [x] [serial] B1 Add `pkgs/tt-model-owner` with the five subcommands, the test-flag gate on every override, the recorded-helper pointer, the unit-name guard, and the command echo. r[tenstorrent.runtime_profile.owner_helper]
- [x] [serial] B2 Add `pkgs/tt-model-owner/check.nix`, which runs the built helper against fake tools and asserts that the helper names the device and reset strings while `tt_model.py` does not. r[tenstorrent.runtime_profile.owner_helper.valid] r[tenstorrent.runtime_profile.owner_helper.override] r[tenstorrent.runtime_profile.owner_helper.unrecorded]
- [x] [serial] B3 Export both tools from the overlay so the NixOS module can reference them without a repository path. r[tenstorrent.runtime_profile.owner_helper]

## Phase 2: Grant

- [x] [serial] G1 Add `hardware.tenstorrent.ttModel` with `operatorUser` and `ownerPackage`, render one NOPASSWD rule, and add the pointer and directory tmpfiles rules. r[tenstorrent.runtime_profile.owner_helper.grant]
- [x] [serial] G2 Add the three evaluation-time assertions and the unit-projection equality check. r[tenstorrent.runtime_profile.owner_helper.grant.valid] r[tenstorrent.runtime_profile.owner_helper.grant.wide]
- [x] [serial] G3 Add `checks/tt-model-nixos-sudo.nix` with the wide-grant, second-rule, and ALL-user negative evaluations. r[tenstorrent.runtime_profile.owner_helper.grant.wide]

## Phase 3: Run

- [x] [serial] R1 Add `tt-model run <profile> [-- <args>]`: run lock, check, card holders, stop, reset, exec with the profile environment, capture the status and receipt, and write the event. r[tenstorrent.runtime_profile.run] r[tenstorrent.runtime_profile.run.valid]
- [x] [serial] R2 Refuse a serve profile, a failed check, a foreign holder, a missing or failing helper, a held lock, and an unattended force. r[tenstorrent.runtime_profile.run.serve] r[tenstorrent.runtime_profile.run.check] r[tenstorrent.runtime_profile.run.holder] r[tenstorrent.runtime_profile.run.helper] r[tenstorrent.runtime_profile.run.lock] r[tenstorrent.runtime_profile.run.unattended]
- [x] [serial] R3 Report the last run in status without letting it affect the verdict. r[tenstorrent.runtime_profile.status.run] r[tenstorrent.runtime_profile.status.run.valid]

## Phase 4: Verification

- [x] [serial] V1 Build the helper check and the `tt-model` install check with every refusal and the run lifecycle. r[tenstorrent.runtime_profile.owner_helper.valid] r[tenstorrent.runtime_profile.run.valid]  (evidence/run-lifecycle.md)
- [x] [serial] V2 Record the sandbox evidence for the helper boundary and the command echo. r[tenstorrent.runtime_profile.owner_helper]  (evidence/helper-boundary.md)
- [x] [serial] V3 Evaluate the sudo check, including the three negative cases and the unchanged unit. r[tenstorrent.runtime_profile.owner_helper.grant.valid] r[tenstorrent.runtime_profile.owner_helper.grant.wide]  (evidence/sudo-grant.md)
- [x] [serial] V4 Run the host refusals that stop before any privileged command. r[tenstorrent.runtime_profile.run.serve] r[tenstorrent.runtime_profile.owner_helper.unrecorded]  (evidence/helper-boundary.md)
- [x] [serial] V5 Build the registry and sudo checks, regenerate the manifest, and run the aggregate quality gate. r[tenstorrent.runtime_profile.owner_helper.grant]  (evidence/quality.md)

## Verification Coverage

- `Scenario: The sandbox exercises the helper without a device` -> B2, V1, V2
- `Scenario: An override is set without the test flag` -> B1, B2
- `Scenario: The host has not recorded this helper` -> B1, B2, V4
- `Scenario: A unit name is not declared` -> B1
- `Scenario: The module renders the intended grant` -> G1, G2, V3
- `Scenario: A wider grant is configured` -> G2, G3, V3
- `Scenario: A job profile runs and is recorded` -> R1, V1
- `Scenario: The profile is a serve profile` -> R2, V1, V4
- `Scenario: The check fails` -> R2, V1
- `Scenario: A card is held outside the owner` -> R2, V1
- `Scenario: The helper is absent or fails` -> R2, V1
- `Scenario: Another run holds the lock` -> R2, V1
- `Scenario: A forced run is unattended` -> R2, V1
- `Scenario: A run event is reported` -> R3, V1
