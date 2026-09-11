# Qwen3.8-27B P150x2 Tuning Admission Boundary Specification Delta

## ADDED Requirements

### Requirement: TP2 tuning selects no winner from incomplete sweep evidence
r[tenstorrent.qwen38_p150x2.tp2_tuning_non_admission] The repository MUST keep the current Qwen3.8 TP2 configurations and safe fallbacks when no complete attributed baseline profile and comparable legal sweep exist. Static shape analysis, another mesh size, compatibility behavior, simulator results, or isolated operation data MUST NOT imply a TP2-tuned configuration or model speedup. Future TP2 tuning admission MUST use a new validated Cairn change with profile-first and matched end-to-end physical evidence.

#### Scenario: TP2 sweep evidence is incomplete
r[tenstorrent.qwen38_p150x2.tp2_tuning_non_admission.valid]
- GIVEN baseline attribution or any required comparable candidate result is missing
- WHEN deterministic TP2 winner selection runs
- THEN it MUST select no winner
- AND current configurations and safe fallbacks MUST remain selected.

#### Scenario: An isolated operation result becomes available
r[tenstorrent.qwen38_p150x2.tp2_tuning_non_admission.invalid]
- GIVEN one operation result is faster without complete model evidence
- WHEN an operator requests TP2 tuning admission
- THEN that isolated result MUST NOT imply a model speedup or generated winner set
- AND a new validated change with matched end-to-end evidence MUST be required.
