# Qwen3.8-27B P150x2 Low-Precision Admission Boundary Specification Delta

## ADDED Requirements

### Requirement: BF4 weight candidates remain experimental without complete admission evidence
r[tenstorrent.qwen38_p150x2.low_precision_non_admission] The repository MUST keep BFP8 as the production precision for the Qwen3.8 weight families covered by the existing opt-in BF4 gates until one named candidate has complete numerical and matched physical evidence. An environment flag, supported cache format, PCC result, static byte count, simulator result, or incomplete receipt MUST NOT imply production admission. Future admission MUST use a new validated Cairn change.

#### Scenario: Candidate evidence is incomplete
r[tenstorrent.qwen38_p150x2.low_precision_non_admission.valid]
- GIVEN a BF4 candidate lacks any required operator, logit, recurrent-state, token-parity, identity, or physical metric
- WHEN production precision admission is evaluated
- THEN the candidate MUST remain unadmitted
- AND BFP8 MUST remain the production profile.

#### Scenario: An experimental BF4 gate exists
r[tenstorrent.qwen38_p150x2.low_precision_non_admission.invalid]
- GIVEN an opt-in BF4 gate or cache format exists in the package
- WHEN an operator requests production admission
- THEN that implementation fact MUST NOT imply correctness, quality, or speed
- AND a new validated change with complete evidence MUST be required.
