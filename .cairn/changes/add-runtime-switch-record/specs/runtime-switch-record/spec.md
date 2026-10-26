## ADDED Requirements

### Requirement: A profile can be checked against one exact package

r[tenstorrent.runtime_profile.check]

`check` MUST validate one profile against one candidate package without writing
anything, opening a device, or needing privilege. It MUST name the exact failing
item and path for every refusal. It MUST validate the profile's existence and
mode, the package's store-path shape, existence, registration and store-name
suffix, the entry binary, the runtime root and execution identity, every
admission reference, the checkpoint and its revision, and any model-target
identity.

#### Scenario: A candidate satisfies the profile

r[tenstorrent.runtime_profile.check.valid]

- **GIVEN** a profile whose package, entry binary, runtime root, admission
  references, checkpoint and model-target identity all match
- **WHEN** `check` runs
- **THEN** it MUST report `pass`, exit zero, and record the registration method
  it used.

#### Scenario: The profile is unknown

r[tenstorrent.runtime_profile.check.unknown_profile]

- **GIVEN** a name that is not in the registry
- **WHEN** `check` runs
- **THEN** it MUST fail with `no such runtime profile`.

#### Scenario: The candidate is not a store path

r[tenstorrent.runtime_profile.check.package_shape]

- **GIVEN** a package argument outside `/nix/store`
- **WHEN** `check` runs
- **THEN** it MUST fail with `not a /nix/store path`.

#### Scenario: The candidate is missing

r[tenstorrent.runtime_profile.check.package_missing]

- **GIVEN** a store path that does not exist
- **WHEN** `check` runs
- **THEN** it MUST fail with `package is missing`.

#### Scenario: The candidate is not registered

r[tenstorrent.runtime_profile.check.package_unregistered]

- **GIVEN** a registration query that runs and fails
- **WHEN** `check` runs
- **THEN** it MUST fail with `package is not registered`.

#### Scenario: The candidate has the wrong store name

r[tenstorrent.runtime_profile.check.store_name]

- **GIVEN** a package whose store name does not end with the profile's suffix
- **WHEN** `check` runs
- **THEN** it MUST fail naming the required suffix.

#### Scenario: The entry binary is absent

r[tenstorrent.runtime_profile.check.entry_binary]

- **GIVEN** a package without the profile's entry binary
- **WHEN** `check` runs
- **THEN** it MUST fail with `entry binary is missing or not executable`.

#### Scenario: The runtime root or execution identity is absent

r[tenstorrent.runtime_profile.check.runtime]

- **GIVEN** a package without the declared runtime root, or without a resolving
  execution-identity symlink
- **WHEN** `check` runs
- **THEN** it MUST fail naming the missing root or identity.

#### Scenario: An admission receipt is absent

r[tenstorrent.runtime_profile.check.receipt_absent]

- **GIVEN** a declared receipt file that does not exist
- **WHEN** `check` runs
- **THEN** it MUST fail with `receipt file is missing`.

#### Scenario: An admission receipt has changed

r[tenstorrent.runtime_profile.check.receipt_digest]

- **GIVEN** a receipt whose bytes do not hash to the recorded digest
- **WHEN** `check` runs
- **THEN** it MUST fail with `receipt bytes hash to`.

#### Scenario: An admission receipt is no longer admitted

r[tenstorrent.runtime_profile.check.receipt_unadmitted]

- **GIVEN** a receipt whose own verdict is not admitted
- **WHEN** `check` runs
- **THEN** it MUST fail with `receipt no longer reports an admitted verdict`.

#### Scenario: An admission receipt names another package

r[tenstorrent.runtime_profile.check.receipt_package]

- **GIVEN** an admitted receipt for a different package
- **WHEN** `check` runs
- **THEN** it MUST fail with `no receipt names this package`.

#### Scenario: An authority document has changed

r[tenstorrent.runtime_profile.check.authority_digest]

- **GIVEN** a generated authority document that does not hash to the recorded
  digest
- **WHEN** `check` runs
- **THEN** it MUST fail with `authority document hashes to`.

#### Scenario: A gate artifact has changed or is absent

r[tenstorrent.runtime_profile.check.gate_artifact]

- **GIVEN** a gate whose recorded artifact is missing or hashes differently
- **WHEN** `check` runs
- **THEN** it MUST fail naming the artifact.

#### Scenario: The checkpoint is absent or misnamed

r[tenstorrent.runtime_profile.check.checkpoint]

- **GIVEN** a checkpoint path that does not exist, or a directory whose name does
  not carry the recorded revision
- **WHEN** `check` runs
- **THEN** it MUST fail with `checkpoint path is missing` or name the revision.

#### Scenario: The model-target identity moved

r[tenstorrent.runtime_profile.check.model_target]

- **GIVEN** a model-target identity that differs from the committed projection
- **WHEN** `check` runs
- **THEN** it MUST fail with the live and expected identities.

#### Scenario: A deep re-hash exceeds the bound

r[tenstorrent.runtime_profile.check.deep.bound]

- **GIVEN** a checkpoint larger than 1073741824 bytes
- **WHEN** `check --deep` runs without `--allow-large`
- **THEN** it MUST refuse before hashing and report the measured size and the
  bound.

#### Scenario: A deep re-hash is bounded and bounded-progress

r[tenstorrent.runtime_profile.check.deep.valid]

- **GIVEN** a checkpoint within the bound
- **WHEN** `check --deep` runs
- **THEN** it MUST hash every file, print progress to standard error, and report
  the file count, byte count and tree digest.

### Requirement: The active record binds the full identity tuple and is rooted

r[tenstorrent.runtime_profile.record]

`record` MUST write `active.json`, an append-only history entry, and one GC root
per store path the record names, atomically and inside the declared roots. The
record MUST carry the profile name, profile schema, mode, package path and store
name, package execution identity, runtime path, checkpoint path and authority,
receipt references with their admission verdicts, gate references, argv and its
BLAKE3, environment, port, reset policy, `admission_bypassed`, `previous`,
`recorded_at`, `recorded_by`, and the registry digest.

#### Scenario: A serve profile is recorded

r[tenstorrent.runtime_profile.record.valid]

- **GIVEN** a serve profile whose check passes, or whose admission failures were
  explicitly forced
- **WHEN** `record` runs
- **THEN** it MUST write the record, the history entry, and the roots
- **AND** `status` MUST report `managed`.

#### Scenario: The profile is a job

r[tenstorrent.runtime_profile.record.job]

- **GIVEN** a `job` profile
- **WHEN** `record` runs
- **THEN** it MUST refuse: the active owner must be a serve profile.

#### Scenario: The check fails structurally

r[tenstorrent.runtime_profile.record.failing_check]

- **GIVEN** a check failure that is not an admission failure
- **WHEN** `record` runs, with or without `--force`
- **THEN** it MUST refuse and name the failing items.

#### Scenario: Only admission failed and the caller forced it

r[tenstorrent.runtime_profile.record.force]

- **GIVEN** only admission failures and `--force`
- **WHEN** `record` runs
- **THEN** it MUST write the record with `admission_bypassed: true`.

#### Scenario: A forced record is unattended

r[tenstorrent.runtime_profile.record.unattended]

- **GIVEN** `TT_MODEL_UNATTENDED=1` and `--force`
- **WHEN** `record` runs
- **THEN** it MUST refuse the force.

#### Scenario: Another writer holds the lock

r[tenstorrent.runtime_profile.record.lock]

- **GIVEN** a held record lock in the same state root
- **WHEN** `record` runs
- **THEN** it MUST refuse with `another record operation holds the lock`.

#### Scenario: The roots cannot be written

r[tenstorrent.runtime_profile.record.roots]

- **GIVEN** a GC-root path that is a symlink or a non-directory, or that cannot
  be created
- **WHEN** `record` runs
- **THEN** it MUST refuse before replacing the active record.

### Requirement: Status proves the record before trusting it

r[tenstorrent.runtime_profile.status.record]

`status` MUST report `managed` only when the record's package exists, its store
name matches the profile, and every store path the record names resolves from
its GC root. It MUST report a legacy source that disagrees with the record as a
contradiction and exit non-zero.

#### Scenario: A rooted record is managed

r[tenstorrent.runtime_profile.status.record.managed]

- **GIVEN** a record whose package and roots all resolve
- **WHEN** `status` runs
- **THEN** it MUST report `managed` and exit zero.

#### Scenario: A root is missing or a package is gone

r[tenstorrent.runtime_profile.status.record.unrooted]

- **GIVEN** a deleted GC root or a store path the record names that no longer
  exists
- **WHEN** `status` runs
- **THEN** it MUST report the root or path and exit non-zero.

#### Scenario: A legacy source disagrees

r[tenstorrent.runtime_profile.status.record.contradiction]

- **GIVEN** a record and an operator symlink or Nix profile that names another
  package
- **WHEN** `status` runs
- **THEN** it MUST report the contradiction and exit non-zero.
