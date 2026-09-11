# RWKV-7 P150x2 Runtime Delta

## ADDED Requirements

### Requirement: Full-device replicated runtime
r[tenstorrent.rwkv7_p150x2.full_device_runtime] The package set MUST provide a pinned RWKV-7 runtime that executes the complete model through TT-NN on an exact replicated `1x2` mesh of physical devices 0 and 1.

#### Scenario: Package boundary contains the complete model path
r[tenstorrent.rwkv7_p150x2.full_device_runtime.complete_path]
- GIVEN the pinned RWKV-7 checkpoint and package-matched TT-Metal runtime
- WHEN the full-device runtime package is built
- THEN its source contains embedding, all twelve layers, recurrent state, final normalization, language-model head, and greedy selection device operations
- AND its production closure contains no CPU model or PyTorch fallback dependency

#### Scenario: Invalid mesh evidence fails
r[tenstorrent.rwkv7_p150x2.full_device_runtime.invalid_mesh]
- GIVEN runtime evidence that omits a card or names an unexpected physical device
- WHEN the evidence classifier validates the runtime receipt
- THEN validation fails with a device-placement diagnostic

### Requirement: Fail-closed evidence classifier
r[tenstorrent.rwkv7_p150x2.evidence_classifier] The package set MUST provide a deterministic classifier that withholds generated text unless complete parity, retained-state, per-card placement, profiler, Inspector, and safety evidence pass.

#### Scenario: Complete evidence passes
r[tenstorrent.rwkv7_p150x2.evidence_classifier.complete]
- GIVEN complete runtime, reference, Inspector, profiler, artifact, and safety evidence
- WHEN the classifier validates the evidence set
- THEN it emits a passing deterministic classification and generated text

#### Scenario: Missing evidence fails
r[tenstorrent.rwkv7_p150x2.evidence_classifier.missing]
- GIVEN an evidence set with a required artifact, card attribution, or safety fact missing
- WHEN the classifier validates the evidence set
- THEN it fails without publishing generated text

### Requirement: Exclusive profiler environment
r[tenstorrent.rwkv7_p150x2.profiler_isolation] A profiling process MUST remove `TT_METAL_WATCHER` and `TT_METAL_DPRINT_CORES` from its environment, and the runtime MUST reject either variable when present with any value.

#### Scenario: Clean profiler environment passes
r[tenstorrent.rwkv7_p150x2.profiler_isolation.clean]
- GIVEN Inspector and the device profiler are enabled
- AND Watcher and kernel DPrint variables are absent
- WHEN the bounded command constructs the device process
- THEN the profiler environment policy passes validation

#### Scenario: Zero-valued Watcher fails
r[tenstorrent.rwkv7_p150x2.profiler_isolation.watcher_zero]
- GIVEN `TT_METAL_DEVICE_PROFILER=1`
- AND `TT_METAL_WATCHER=0` is present
- WHEN the profiler environment is validated
- THEN validation fails before a device opens

#### Scenario: Present DPrint fails
r[tenstorrent.rwkv7_p150x2.profiler_isolation.dprint_present]
- GIVEN `TT_METAL_DEVICE_PROFILER=1`
- AND `TT_METAL_DPRINT_CORES` is present with an empty or non-empty value
- WHEN the profiler environment is validated
- THEN validation fails before a device opens

### Requirement: Device-free validation boundary
r[tenstorrent.rwkv7_p150x2.device_free_validation] The repository MUST build and test the runtime, classifier, environment policy, source boundary, and production closure without acquiring a Tenstorrent device.

#### Scenario: Focused checks run without hardware
r[tenstorrent.rwkv7_p150x2.device_free_validation.focused]
- GIVEN no authorization to stop owner services or open device nodes
- WHEN the focused Nix checks run
- THEN all package and policy checks execute without a hardware process

#### Scenario: CPU fallback enters closure
r[tenstorrent.rwkv7_p150x2.device_free_validation.cpu_fallback]
- GIVEN a production closure that contains a CPU model or PyTorch fallback dependency
- WHEN the closure check runs
- THEN the check fails

### Requirement: Attempt-3 retention
r[tenstorrent.rwkv7_p150x2.attempt3_retention] The repository MUST retain a bounded immutable fixture that classifies attempt 3 as a one-process BRISC firmware overflow after both physical devices opened.

#### Scenario: Recorded failure class passes
r[tenstorrent.rwkv7_p150x2.attempt3_retention.recorded]
- GIVEN the retained attempt-3 fixture
- WHEN the retained-evidence check runs
- THEN it confirms devices 0 and 1 opened
- AND it confirms profiler and Watcher firmware macros appeared together
- AND it confirms BRISC overflowed before verified generation

#### Scenario: Success claim is injected
r[tenstorrent.rwkv7_p150x2.attempt3_retention.false_success]
- GIVEN a modified attempt-3 fixture that claims verified success
- WHEN the retained-evidence check runs
- THEN the check fails

### Requirement: No hardware claim
r[tenstorrent.rwkv7_p150x2.no_hardware_claim] This change MUST NOT claim fresh two-card generation, performance, or physical placement from device-free validation.

#### Scenario: Device-free change completes
r[tenstorrent.rwkv7_p150x2.no_hardware_claim.complete]
- GIVEN all device-free checks pass
- WHEN this change is archived
- THEN the accepted specification states that a fresh machine-bound attempt is still required
