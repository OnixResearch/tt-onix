## ADDED Requirements

### Requirement: One registry for every switchable runtime

r[tenstorrent.runtime_profile.registry]

The repository MUST record every switchable Tenstorrent model runtime in one
typed registry. A profile MUST name the package attribute and store-name suffix,
the entry binary and its exact argument prefix, the launcher environment, the
mesh, the limits, the checkpoint and its authority, the cache, the precision
selectors, the admission kind, the readiness signal, the reset policy, and its
non-claims. Every value MUST be transcribed from a source in this tree or a
recorded receipt, and the profile MUST cite that source. The registry MUST NOT
evaluate an admission verdict or re-serialize a model's receipt schema.

#### Scenario: Every switchable runtime has exactly one profile

r[tenstorrent.runtime_profile.registry.valid]

- **GIVEN** the four runtimes this repository ships on two P150 cards
- **WHEN** the registry is read
- **THEN** it MUST carry `qwen38`, `nanbeige43`, `rwkv7-ttnn`, and
  `metalium-rs-rwkv7`, once each
- **AND** every profile MUST name the source its values came from.

#### Scenario: A profile name is duplicated

r[tenstorrent.runtime_profile.registry.unique]

- **GIVEN** a registry with two profiles of the same name
- **WHEN** the registry contract is applied
- **THEN** validation MUST fail and name the uniqueness rule.

### Requirement: The profile contract is closed and typed

r[tenstorrent.runtime_profile.contract]

The profile MUST be a closed record contract. An unknown field, a missing
required field, a value outside a closed set, or a malformed digest MUST fail
validation with a message that names the rule. Cross-field rules MUST decide
when an optional field is required or forbidden.

#### Scenario: An undocumented field is added

r[tenstorrent.runtime_profile.contract.closed]

- **GIVEN** a profile with a field the contract does not declare
- **WHEN** the registry contract is applied
- **THEN** validation MUST fail and name the extra field.

#### Scenario: A required field is removed

r[tenstorrent.runtime_profile.contract.required]

- **GIVEN** a profile without a required field
- **WHEN** the registry contract is applied
- **THEN** validation MUST fail and name the missing field.

#### Scenario: A closed set is extended by hand

r[tenstorrent.runtime_profile.contract.mode]

- **GIVEN** a profile whose `mode` is neither `serve` nor `job`
- **WHEN** the registry contract is applied
- **THEN** validation MUST fail with the mode rule.

#### Scenario: A digest is not lowercase hexadecimal

r[tenstorrent.runtime_profile.contract.hash]

- **GIVEN** an authority digest in uppercase or with the wrong length
- **WHEN** the registry contract is applied
- **THEN** validation MUST fail at the digest field.

#### Scenario: A serve profile does not own the admitted mesh

r[tenstorrent.runtime_profile.contract.serve_mesh]

- **GIVEN** a serve profile whose physical device ids are not exactly `[0, 1]`
- **WHEN** the registry contract is applied
- **THEN** validation MUST fail with the serve-mesh rule.

#### Scenario: A serve profile has no port

r[tenstorrent.runtime_profile.contract.serve_port]

- **GIVEN** a serve profile without a port
- **WHEN** the registry contract is applied
- **THEN** validation MUST fail with the serve-port rule.

### Requirement: The committed registry is fresh and its references are true

r[tenstorrent.runtime_profile.registry.fresh]

The committed export MUST be byte-identical to its Nickel source. Every recorded
authority digest MUST be the BLAKE3 of the document the profile names. Every
recorded receipt MUST exist, MUST hash to the recorded value, and MUST still
report an admitted verdict. Every `model_target_blake3` MUST equal the live
identity the model-target registry holds. The check MUST be registered in the
closed required-check catalog.

#### Scenario: The registry is fresh and every reference resolves

r[tenstorrent.runtime_profile.registry.fresh.valid]

- **GIVEN** the committed export, the authority documents, and the receipts
- **WHEN** the registry check runs
- **THEN** regeneration MUST equal the committed export
- **AND** every recorded digest MUST match its file
- **AND** every referenced receipt MUST still be admitted.

#### Scenario: The export or a reference drifts

r[tenstorrent.runtime_profile.registry.fresh.drift]

- **GIVEN** a stale export, an edited authority document, a superseded receipt,
  or a moved model-target identity
- **WHEN** the registry check runs
- **THEN** it MUST fail and name the file or identity that moved
- **AND** a negative fixture MUST be refused with its own rule.

### Requirement: Status is read-only and names what is broken

r[tenstorrent.runtime_profile.status]

A status command MUST report the active runtime and every legacy source it can
read, without root, sudo, a device open, a write, or a unit mutation. It MUST
report the root-owned record when one exists and otherwise the operator symlink,
the Nix profile generation, and the unit state. It MUST degrade to `unavailable`
for a source it cannot read. It MUST exit non-zero when a reported source is
dangling, unrooted, or contradicts another source, and print that path and
reason. It MUST treat a record that names an unknown profile as a broken source,
and MUST exit as a usage error, not a guess, for a registry that does not match
its contract.

#### Scenario: A rooted record is the active runtime

r[tenstorrent.runtime_profile.status.managed]

- **GIVEN** a record whose package exists, matches its profile's store name, and
  resolves from its GC root
- **WHEN** status runs
- **THEN** it MUST report `managed` and exit zero
- **AND** it MUST report a legacy source that disagrees with the record as a
  contradiction and exit non-zero.

#### Scenario: A recorded package or a legacy pointer is gone

r[tenstorrent.runtime_profile.status.dangling]

- **GIVEN** a record package, a GC root, an operator symlink, or a profile
  generation whose target no longer exists
- **WHEN** status runs
- **THEN** it MUST report `broken`, exit non-zero, and name the path and reason.

#### Scenario: Nothing claims the cards

r[tenstorrent.runtime_profile.status.unmanaged]

- **GIVEN** no record and no resolving legacy source
- **WHEN** status runs
- **THEN** it MUST report `none`, or `degraded` when a source was unavailable,
  and exit zero.

#### Scenario: The registry does not match its contract

r[tenstorrent.runtime_profile.status.invalid]

- **GIVEN** a registry export with the wrong schema version or a duplicate
  profile name, or a record naming an unknown profile
- **WHEN** status runs
- **THEN** it MUST refuse with the specific mismatch instead of guessing.
