## ADDED Requirements

### Requirement: A serve profile becomes the verified owner in one sequence

r[tenstorrent.runtime_profile.switch]

`switch` MUST accept only `mode = serve` profiles. It MUST take the switch lock,
check the candidate package, refuse a foreign card holder, stop a live owner,
reset when the profile or the stopped owner requires it, check the candidate
again, run the profile's containment check when one is declared, commit the
candidate record and every GC root, start the owner, wait for readiness, read the
instance's identity back, and commit the verified verdict.

#### Scenario: A healthy switch commits a verified record

r[tenstorrent.runtime_profile.switch.valid]

- **GIVEN** a serve profile whose check passes, no foreign holder, and an instance
  that becomes ready and reports the expected identity
- **WHEN** `switch` runs
- **THEN** it MUST write the candidate record, start the owner, record the
  readiness time, compare the identity, and replace the record with the verified
  one
- **AND** a `.verified` history entry MUST exist beside the candidate entry.

#### Scenario: The profile is a job

r[tenstorrent.runtime_profile.switch.mode]

- **GIVEN** a `job` profile
- **WHEN** `switch` runs
- **THEN** it MUST refuse and point at `tt-model run`.

#### Scenario: The check fails

r[tenstorrent.runtime_profile.switch.check]

- **GIVEN** a structural failure, with or without `--force`, or an admission
  failure without `--force`
- **WHEN** `switch` runs
- **THEN** it MUST refuse and name the failing items.

#### Scenario: A card is held outside the owner

r[tenstorrent.runtime_profile.switch.holder]

- **GIVEN** a holder whose unit differs from the declared owner
- **WHEN** `switch` runs
- **THEN** it MUST refuse before stopping anything
- **AND** the helper MUST NOT receive `stop-owner`.

#### Scenario: The helper is absent, fails, or the start fails

r[tenstorrent.runtime_profile.switch.helper]

- **GIVEN** no executable helper, a `cards` failure, or a `start-owner` failure
- **WHEN** `switch` runs
- **THEN** it MUST refuse and report the helper path and status.

#### Scenario: Readiness does not arrive

r[tenstorrent.runtime_profile.switch.readiness]

- **GIVEN** an instance that never answers within the profile's timeout
- **WHEN** `switch` runs
- **THEN** it MUST refuse with `readiness` and the observed timeout.

#### Scenario: The instance reports another identity

r[tenstorrent.runtime_profile.switch.identity]

- **GIVEN** a health or evidence payload whose package identity, model identity,
  device ids, `simulated`, runtime identity, or profile id does not match the
  record
- **WHEN** `switch` runs
- **THEN** it MUST refuse and name each mismatched field.

#### Scenario: Containment fails

r[tenstorrent.runtime_profile.switch.containment]

- **GIVEN** a declared containment command whose verdict is not the declared one
- **WHEN** `switch` runs, with or without `--force`
- **THEN** it MUST refuse before writing the record.

#### Scenario: Another switch holds the lock

r[tenstorrent.runtime_profile.switch.lock]

- **GIVEN** a held switch lock in the same state root
- **WHEN** `switch` runs
- **THEN** it MUST refuse with the lock path.

#### Scenario: A forced switch is unattended

r[tenstorrent.runtime_profile.switch.unattended]

- **GIVEN** `TT_MODEL_UNATTENDED=1` and `--force` with an admission failure
- **WHEN** `switch` runs
- **THEN** it MUST refuse the force.

#### Scenario: Only admission failed and the caller forced it

r[tenstorrent.runtime_profile.switch.force]

- **GIVEN** only admission failures and `--force`
- **WHEN** `switch` runs
- **THEN** it MUST proceed and record `admission_bypassed: true`.

### Requirement: A failed switch restores what it replaced

r[tenstorrent.runtime_profile.switch.rollback]

On any failure after the candidate record was committed, the switch MUST restore
the previous record byte for byte and, when an owner was running, start it again
and verify its readiness and identity. When that verification fails, it MUST
withdraw the restored record, keep it in history, and write the failure details.

#### Scenario: A failed switch leaves the previous record unchanged

r[tenstorrent.runtime_profile.switch.rollback.valid]

- **GIVEN** a committed record and a switch that fails during start, readiness, or
  identity
- **WHEN** the switch rolls back
- **THEN** `active.json` MUST be byte-identical to the previous record
- **AND** `switch-failure.json` MUST record `restored: true` and the reason.

#### Scenario: The previous owner cannot be restored

r[tenstorrent.runtime_profile.switch.rollback.failed]

- **GIVEN** a previous owner that was running and a restart that fails
- **WHEN** the switch rolls back
- **THEN** `active.json` MUST be removed, the restored record MUST be kept as a
  history entry, and the failure MUST be reported with `restored: false`.

### Requirement: The host fragment is reviewable and renders one grant

r[tenstorrent.runtime_profile.switch.host]

The repository MUST ship the host activation as an example module that is
imported by nothing, together with an evaluation check proving it adds exactly
one unit, renders exactly one sudo rule for the owner helper, keeps the reviewed
serving options, and rejects a wider grant.

#### Scenario: The fragment renders the intended host

r[tenstorrent.runtime_profile.switch.host.valid]

- **GIVEN** the example module
- **WHEN** the configuration is evaluated
- **THEN** exactly `qwen38-p150x2` MUST be added to `systemd.services`
- **AND** exactly one NOPASSWD rule for `${helper}/bin/tt-model-owner` MUST exist
- **AND** the unit MUST keep the pinned model path, the 131072-token row, the
  eight-slot scheduler, sampled fusion, and the declared competing owners.

#### Scenario: A wider grant is configured

r[tenstorrent.runtime_profile.switch.host.wide]

- **GIVEN** a rule granting the helper's `bin` directory
- **WHEN** the configuration is evaluated
- **THEN** an assertion MUST fail and name the reason.
