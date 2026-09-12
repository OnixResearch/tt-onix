## ADDED Requirements

### Requirement: The served Qwen3.8 artifact matches a measured execution identity

r[tenstorrent.qwen38_p150x2.runtime_deployment]

The repository MUST NOT leave the served Qwen3.8 service on an execution
identity that no held receipt describes without recording the mismatch as a
deployment gap. Before a switch, a readiness gate MUST start the exact serving
artifact the target configuration selects, wait for readiness, serve one greedy
request, and record the artifact's package identity, runtime identity, and
response. After a switch, readiness, health, the served identity, and the
single-stream gate status MUST be re-confirmed, and the batch-serving route MUST
remain disabled unless a receipt admits it. The switch MUST NOT move any
admission binding and MUST NOT claim that the runtime change itself is admitted
by any batch receipt.

#### Scenario: The served artifact matches a measured execution identity

r[tenstorrent.qwen38_p150x2.runtime_deployment.valid]

- **GIVEN** a target configuration selecting a package and runtime
- **WHEN** the readiness gate runs the serving artifact on the isolated devices
- **THEN** the artifact MUST report ready
- **AND** one greedy request MUST return the admitted token prefix
- **AND** the recorded package and runtime identities MUST name the artifact that
  the switch will deploy.

#### Scenario: The switch carries no new admission

r[tenstorrent.qwen38_p150x2.runtime_deployment.invalid]

- **GIVEN** an admitted batch profile for one width
- **WHEN** the service is configured for production serving
- **THEN** the batch-admission route MUST stay disabled
- **AND** no response MAY be labelled admission-eligible by the runtime change
  alone.

#### Scenario: The shared runtime and its consumers are untouched

r[tenstorrent.qwen38_p150x2.runtime_deployment.unchanged]

- **GIVEN** the shared tt-metal 0.77 output consumed by RWKV-7 and the training
  platform
- **WHEN** the service moves onto a measured identity
- **THEN** the shared output path MUST be unchanged
- **AND** the RWKV-7 and training checks MUST still pass.
