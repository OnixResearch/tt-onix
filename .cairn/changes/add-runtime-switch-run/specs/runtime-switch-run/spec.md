## ADDED Requirements

### Requirement: The privileged operations live in one helper

r[tenstorrent.runtime_profile.owner_helper]

The repository MUST keep every device, reset, unit, and root-write operation in
one privileged helper with a closed subcommand set: `cards`, `stop-owner`,
`start-owner`, `reset`, `record`. The device-free tool MUST NOT name a device
node, the reset tool, or the legacy owner-control helper, and the helper MUST
name them. The helper MUST print each acting command before running it, MUST
refuse any overridden state root, GC-root directory, unit name, or command path
unless `TT_MODEL_OWNER_TEST=1`, MUST refuse unless the host's recorded helper
pointer resolves to its own store path, and MUST NOT guess a unit name.

#### Scenario: The sandbox exercises the helper without a device

r[tenstorrent.runtime_profile.owner_helper.valid]

- **GIVEN** the built helper and fake `systemctl`, `lsof`, `tt-smi`, and record
  tools
- **WHEN** the helper runs `cards`, `reset`, `stop-owner`, or `record`
- **THEN** it MUST report the owner and every holder, print each command it runs,
  and reach the fake tool
- **AND** the device-free tool MUST NOT contain the device or reset strings.

#### Scenario: An override is set without the test flag

r[tenstorrent.runtime_profile.owner_helper.override]

- **GIVEN** a non-default state root, GC-root directory, unit, or command path
  and no `TT_MODEL_OWNER_TEST=1`
- **WHEN** the helper runs
- **THEN** it MUST refuse and name the variable.

#### Scenario: The host has not recorded this helper

r[tenstorrent.runtime_profile.owner_helper.unrecorded]

- **GIVEN** a missing helper pointer, or one that resolves to another store path
- **WHEN** the helper runs any subcommand
- **THEN** it MUST refuse and name the expected and recorded paths.

#### Scenario: A unit name is not declared

r[tenstorrent.runtime_profile.owner_helper.unit]

- **GIVEN** an empty owner unit
- **WHEN** the helper runs
- **THEN** it MUST refuse rather than guess.

### Requirement: The sudo grant is one helper binary and nothing wider

r[tenstorrent.runtime_profile.owner_helper.grant]

The NixOS module MUST render exactly one sudo rule for one configured user, for
the helper binary, with `NOPASSWD`. It MUST assert at evaluation time that the
operator is not `ALL`, that exactly one rule names the helper and grants no other
command, and that no rule for the operator grants anything inside the helper
package other than that binary. It MUST NOT enable, unmask, start, or otherwise
change any existing unit.

#### Scenario: The module renders the intended grant

r[tenstorrent.runtime_profile.owner_helper.grant.valid]

- **GIVEN** `hardware.tenstorrent.ttModel.enable = true`
- **WHEN** the configuration is evaluated
- **THEN** exactly one rule MUST name `${helper}/bin/tt-model-owner` with
  `NOPASSWD` for the configured user
- **AND** the `qwen38-p150x2` unit projection MUST be identical with and without
  the option.

#### Scenario: A wider grant is configured

r[tenstorrent.runtime_profile.owner_helper.grant.wide]

- **GIVEN** a second rule granting the helper's `bin` directory, or the helper
  with additional commands, or `ALL` as the operator user
- **WHEN** the configuration is evaluated
- **THEN** an assertion MUST fail and name the reason.

### Requirement: `run` takes the cards for one job profile

r[tenstorrent.runtime_profile.run]

`run` MUST accept only `mode = job` profiles. It MUST take the run lock, check
the profile against the candidate package, refuse a foreign card holder, stop a
running owner through the helper, reset when the profile or the stopped owner
requires it, run the profile's argv with the profile's environment and no
inherited model identity, and write a run event. It MUST NOT write
`active.json`.

#### Scenario: A job profile runs and is recorded

r[tenstorrent.runtime_profile.run.valid]

- **GIVEN** a job profile whose check passes, an idle or owned board, and the
  owner helper
- **WHEN** `run` executes
- **THEN** it MUST stop the owner, reset when required, run the argv, and write
  `last-run.json` plus a run event with profile, package, argv and digest,
  receipt, exit status, timestamps, and the owner and reset flags
- **AND** `active.json` MUST remain absent.

#### Scenario: The profile is a serve profile

r[tenstorrent.runtime_profile.run.serve]

- **GIVEN** a `serve` profile
- **WHEN** `run` executes
- **THEN** it MUST refuse and point at `tt-model switch`.

#### Scenario: The check fails

r[tenstorrent.runtime_profile.run.check]

- **GIVEN** a structural failure, with or without `--force`, or an admission
  failure without `--force`
- **WHEN** `run` executes
- **THEN** it MUST refuse and name the failing items.

#### Scenario: A card is held outside the owner

r[tenstorrent.runtime_profile.run.holder]

- **GIVEN** a holder whose unit differs from the declared owner
- **WHEN** `run` executes
- **THEN** it MUST refuse, naming the device, the process, and the owner unit.

#### Scenario: The helper is absent or fails

r[tenstorrent.runtime_profile.run.helper]

- **GIVEN** no executable helper, or a helper whose `cards` fails
- **WHEN** `run` executes
- **THEN** it MUST refuse and report the helper path and its status.

#### Scenario: Another run holds the lock

r[tenstorrent.runtime_profile.run.lock]

- **GIVEN** a held run lock in the same state root
- **WHEN** `run` executes
- **THEN** it MUST refuse with the lock path.

#### Scenario: A forced run is unattended

r[tenstorrent.runtime_profile.run.unattended]

- **GIVEN** `TT_MODEL_UNATTENDED=1` and `--force` with an admission failure
- **WHEN** `run` executes
- **THEN** it MUST refuse the force.

### Requirement: A run event is reported but never becomes the owner

r[tenstorrent.runtime_profile.status.run]

`status` MUST report the most recent run event and MUST NOT let it change the
active verdict.

#### Scenario: A run event is reported

r[tenstorrent.runtime_profile.status.run.valid]

- **GIVEN** a state root with `last-run.json` and no `active.json`
- **WHEN** `status` runs
- **THEN** it MUST report the last run's profile, package, and exit status
- **AND** it MUST still report the active record as absent.
