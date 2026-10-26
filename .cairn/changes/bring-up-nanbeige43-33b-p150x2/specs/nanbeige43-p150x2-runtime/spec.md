## ADDED Requirements

### Requirement: The pinned checkpoint is admitted before device work

r[tenstorrent.nanbeige43_p150x2.checkpoint_authority]

The package MUST bind revision `c1093c501cc6460427c0ac634db7223d0450c0ee` and
the SHA-256 identity of every shard and metadata file. Every command that loads
weights MUST verify the checkpoint directory against that authority before it
opens a device. A reduced-layer checkpoint MUST record the identities of the
source tensors it was derived from.

#### Scenario: Matching checkpoint is admitted

r[tenstorrent.nanbeige43_p150x2.checkpoint_authority.valid]

- **GIVEN** the verified download directory
- **WHEN** a command validates it
- **THEN** validation MUST pass without reading device state.

#### Scenario: Drifted checkpoint is refused

r[tenstorrent.nanbeige43_p150x2.checkpoint_authority.invalid]

- **GIVEN** a missing, extra, or altered shard, index, config, or tokenizer file
- **WHEN** a command validates it
- **THEN** the command MUST exit before any device open
- **AND** it MUST name the mismatched file.

### Requirement: Production generation is fail-closed on the exact P150x2 mesh

r[tenstorrent.nanbeige43_p150x2.mesh]

The production command MUST open exactly a `1x2` mesh on physical PCIe devices
0 and 1 with `FABRIC_1D`. It MUST refuse any other device set before it loads
weights. The hardware-neutral `1x1` smoke mode MUST accept only a reduced-layer
checkpoint. It MUST NOT add simulator-specific runtime branches.

#### Scenario: Wrong device set is refused

r[tenstorrent.nanbeige43_p150x2.mesh.invalid]

- **GIVEN** PCIe device IDs other than `[0, 1]`, or a mesh that opens fewer
  than two devices
- **WHEN** the production command starts
- **THEN** it MUST exit before weight loading.

### Requirement: Device output agrees with the CPU reference

r[tenstorrent.nanbeige43_p150x2.reference_agreement]

The device run MUST be compared with a BF16 HF transformers CPU reference of
the same checkpoint and tokenizer. A receipt MUST record runtime, package,
checkpoint, precision preset, mesh, and prompt identities, together with prefill
logits PCC and teacher-forced top-1/top-5 agreement. Simulator receipts MUST be
marked simulated and MUST NOT claim physical execution or performance.

#### Scenario: Full model meets the teacher-forced gate

r[tenstorrent.nanbeige43_p150x2.reference_agreement.valid]

- **GIVEN** the full 52-layer model on the P150x2 mesh with the admitted preset
- **WHEN** 512 teacher-forced decode positions run against the reference
- **THEN** top-1 agreement MUST be at least 87 % and top-5 at least 97 %
- **AND** the receipt MUST record both values and the preset.

#### Scenario: Gate failure is recorded, not admitted

r[tenstorrent.nanbeige43_p150x2.reference_agreement.blocked]

- **GIVEN** a run below either threshold, an unmodeled simulator operation, or a
  device error
- **WHEN** the run ends
- **THEN** the receipt MUST NOT be admitted
- **AND** the exact failing gate or diagnostic MUST be recorded.
