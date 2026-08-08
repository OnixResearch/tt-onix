# Rwkv7 P150x2 Runtime Specification

## Purpose

Defines the `rwkv7-p150x2-runtime` capability.

## Requirements

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

### Requirement: Fast inference mode is separate from evidence mode
r[tenstorrent.rwkv7_p150x2.fast_inference_mode] The runtime MUST provide a named fast inference mode that preserves the existing evidence mode. The fast mode MUST exclude state readback, profiler flush, control replay, and artifact writes from its steady-state decode interval.

#### Scenario: Fast decode excludes evidence synchronization
r[tenstorrent.rwkv7_p150x2.fast_inference_mode.valid]
- GIVEN loaded weights, warmed programs, and initialized device state
- WHEN the fast mode measures steady-state decode
- THEN the timed interval MUST contain token decode work only
- AND compile, warm-up, evidence readback, profiler flush, and teardown MUST be reported outside that interval.

#### Scenario: Evidence work enters the timed interval
r[tenstorrent.rwkv7_p150x2.fast_inference_mode.invalid]
- GIVEN a fast-mode trace that contains a state readback, profiler flush, control replay, or artifact write
- WHEN measurement admission validates the trace
- THEN admission MUST fail and MUST name the forbidden operation.

### Requirement: Fast mode uses independent replicas
r[tenstorrent.rwkv7_p150x2.independent_replicas] The P150x2 fast mode MUST run one independent complete RWKV-7 replica per physical card. Each live request MUST own exactly one card-local recurrent-state slot. The scheduler MUST NOT assign the same live request or state slot to both replicas.

#### Scenario: Distinct requests use both replicas
r[tenstorrent.rwkv7_p150x2.independent_replicas.valid]
- GIVEN at least two ready requests and an admitted `p150_x2` mesh
- WHEN the deterministic scheduler fills both replicas
- THEN physical devices 0 and 1 MUST receive distinct request IDs
- AND each request MUST retain an independent card-local state slot.

#### Scenario: Lockstep ownership is rejected
r[tenstorrent.rwkv7_p150x2.independent_replicas.invalid]
- GIVEN a fast-mode schedule that assigns one request ID or one live state slot to both cards
- WHEN the scheduler core validates the schedule
- THEN validation MUST fail before decode dispatch.

### Requirement: Recurrent state remains device-resident and FP32
r[tenstorrent.rwkv7_p150x2.device_resident_state] The fast runtime MUST retain token-shift and WKV state on the selected device across decode steps. The WKV recurrence MUST use FP32 state. Host transport of complete recurrent state MUST NOT occur between timed decode steps.

#### Scenario: FP32 state remains on device
r[tenstorrent.rwkv7_p150x2.device_resident_state.valid]
- GIVEN an admitted request with initialized FP32 recurrent state
- WHEN two or more fast decode steps execute
- THEN each step MUST consume the prior device state
- AND no complete recurrent-state readback or rewrite MAY occur between those steps.

#### Scenario: BF16 recurrent state is rejected
r[tenstorrent.rwkv7_p150x2.device_resident_state.invalid]
- GIVEN a fast-mode profile or runtime boundary that declares BF16 recurrent state
- WHEN profile or runtime admission runs
- THEN admission MUST fail before physical device execution.

### Requirement: Throughput comparison is attributable and bounded
r[tenstorrent.rwkv7_p150x2.throughput_measurement] A physical throughput comparison MUST bind one-card and two-card runs to the same checkpoint, runtime closure, workload per replica, profile, machine facts, and sampling settings. The typed profile MUST own the minimum dual-card speedup and maximum p99 inter-token latency regression bounds.

#### Scenario: Complete physical comparison passes
r[tenstorrent.rwkv7_p150x2.throughput_measurement.valid]
- GIVEN repeated one-card and two-card physical runs with complete profiler and per-card attribution
- WHEN the deterministic admission core computes aggregate throughput and per-stream p99 latency
- THEN it MUST report the dual-card speedup ratio
- AND it MUST report the p99 latency regression fraction
- AND it MUST classify each named profile bound independently.

#### Scenario: Incomplete or non-finite metrics fail
r[tenstorrent.rwkv7_p150x2.throughput_measurement.invalid]
- GIVEN a comparison with missing card attribution, missing profiler data, or a non-finite metric
- WHEN measurement admission runs
- THEN admission MUST fail
- AND it MUST NOT emit a passing throughput classification.

### Requirement: Simulator smoke precedes physical optimization
r[tenstorrent.rwkv7_p150x2.simulator_gate] A new physical optimization attempt MUST follow a package-matched Blackhole simulator smoke that opens one device, compiles firmware, loads representative weights, executes one RWKV layer, and carries recurrent state. Simulator evidence MUST be labeled as simulated.

#### Scenario: Simulator smoke passes without a performance claim
r[tenstorrent.rwkv7_p150x2.simulator_gate.valid]
- GIVEN the production host and device binaries with packaged `libttsim_bh.so`
- WHEN the bounded `1x1` smoke completes
- THEN its receipt MUST record device open, firmware compilation, representative weight load, one-layer execution, and state carry
- AND the receipt MUST state that it proves no physical placement or performance.

#### Scenario: Simulator evidence claims speedup
r[tenstorrent.rwkv7_p150x2.simulator_gate.invalid]
- GIVEN a simulator receipt that claims physical throughput, latency, placement, or scaling
- WHEN receipt admission runs
- THEN admission MUST fail and MUST classify the claim as unsupported.

### Requirement: Speedup claims require the physical target
r[tenstorrent.rwkv7_p150x2.physical_speedup_claim] The package MUST NOT claim improved P150x2 throughput unless a complete physical receipt meets the profile-owned minimum dual-card speedup and maximum p99 latency regression bounds. The claim MUST apply only to aggregate throughput for the named independent-request workload.

#### Scenario: Physical speedup target passes
r[tenstorrent.rwkv7_p150x2.physical_speedup_claim.valid]
- GIVEN a complete physical comparison whose dual-card speedup and p99 latency regression satisfy both profile bounds
- WHEN the evidence classifier admits the result
- THEN the package MAY publish the aggregate throughput claim with the profile and receipt identities
- AND it MUST retain the single-stream latency non-claim.

#### Scenario: Target miss cannot claim speedup
r[tenstorrent.rwkv7_p150x2.physical_speedup_claim.invalid]
- GIVEN a physical comparison that misses either profile bound or lacks complete attribution
- WHEN the evidence classifier evaluates the result
- THEN it MUST withhold the speedup claim
- AND it MUST retain the measured blocker without publishing generated performance text.
