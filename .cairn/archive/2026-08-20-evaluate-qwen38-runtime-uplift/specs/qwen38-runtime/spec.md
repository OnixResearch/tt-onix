# Qwen3.8 Runtime Admission Boundary Specification Delta

## ADDED Requirements

### Requirement: A runtime uplift remains unadmitted without one complete isolated candidate
r[tenstorrent.qwen38.runtime_uplift_non_admission] The repository MUST keep the current Qwen3.8 runtime and compatibility fallbacks when a runtime-uplift evaluation lacks one immutable isolated candidate. A newer version, public capability name, simulator result, or partial source pin MUST NOT imply compatibility or admission. Future runtime-uplift evaluation MUST use a new validated Cairn change with complete runtime, model, patch, ABI, firmware, driver, topology, capability, and matched physical identities.

#### Scenario: No complete runtime candidate exists
r[tenstorrent.qwen38.runtime_uplift_non_admission.valid]
- GIVEN no isolated candidate binds every required source and machine identity
- WHEN runtime-uplift admission is evaluated
- THEN the current runtime and compatibility fallbacks MUST remain selected
- AND no candidate package, service, or performance claim MUST be added.

#### Scenario: A newer runtime becomes available
r[tenstorrent.qwen38.runtime_uplift_non_admission.invalid]
- GIVEN a newer runtime release or operation name appears after this decision
- WHEN an operator requests runtime admission
- THEN availability or version MUST NOT imply behavioral support
- AND a new validated change with complete capability and physical evidence MUST be required.
