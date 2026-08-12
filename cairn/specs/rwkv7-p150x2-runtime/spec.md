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

### Requirement: Scalable card-local batching
r[tenstorrent.rwkv7_p150x2.scalable_card_local_batching] The fast runtime MUST derive card-local request capacity from a typed profile and a checked memory plan. It MUST preserve unique request state, FP32 recurrence, deterministic scheduling, and exact generated-token parity at every promoted capacity. A capacity above four MUST NOT become the default without stable physical throughput evidence and strict two-card admission.

#### Scenario: Larger capacity passes
r[tenstorrent.rwkv7_p150x2.scalable_card_local_batching.valid]
- GIVEN a supported candidate capacity with sufficient checked device memory
- AND every live request owns one unique zero-initialized card-local state slot
- WHEN simulator, exact-parity, alternating A/B, and strict physical admission complete
- THEN the runtime MAY promote that capacity
- AND the receipt MUST report capacity, effective batch size, memory use, throughput, latency, card attribution, and evidence identities.

#### Scenario: Unsafe capacity fails before device access
r[tenstorrent.rwkv7_p150x2.scalable_card_local_batching.invalid_memory]
- GIVEN a capacity plan with arithmetic overflow, insufficient memory reserve, or an unsupported capacity
- WHEN capacity admission runs
- THEN admission MUST fail before a device opens
- AND it MUST identify the rejected resource bound.

#### Scenario: State ownership is invalid
r[tenstorrent.rwkv7_p150x2.scalable_card_local_batching.invalid_state]
- GIVEN a schedule with duplicate live-slot ownership, stale recycled state, or cross-card request ownership
- WHEN the pure scheduler validates the plan
- THEN validation MUST fail before decode dispatch.

#### Scenario: Throughput target is missed
r[tenstorrent.rwkv7_p150x2.scalable_card_local_batching.target_miss]
- GIVEN a larger capacity that fails exact parity, stability, throughput, or p99 latency admission
- WHEN the bounded sweep evaluates the candidate
- THEN the runtime MUST keep the last admitted capacity
- AND the evidence package MUST retain the measured blocker without claiming the target rate.

### Requirement: Per-token capacity admission
r[tenstorrent.rwkv7_p150x2.scalable_card_local_batching.per_token_admission] The capacity admission gate MUST compare each candidate against the current admitted capacity on p99 time per generated token, computed as the p99 of (sample elapsed divided by tokens served in that sample). Per-request wall latency MUST NOT be the admission basis across different capacity workloads, because it grows with batch concurrency and does not measure per-token device cost. A promoted candidate MUST keep the strict physical, profiler, parity, and occupancy rails from the parent requirement.

#### Scenario: Candidate faster per token is promoted
r[tenstorrent.rwkv7_p150x2.scalable_card_local_batching.per_token_admission.valid]
- GIVEN a candidate measured on physical cards with exact FP32-reference parity and profiler attribution
- AND the candidate's aggregate tokens per second exceeds the current admitted capacity
- AND the candidate's p99 time per generated token does not regress beyond the profile bound
- WHEN the pure gate evaluates the order-balanced evidence
- THEN the capacity MAY be promoted
- AND the evidence package MUST retain the order-balanced runs and the admission decision.

#### Scenario: Metric basis is rejected
r[tenstorrent.rwkv7_p150x2.scalable_card_local_batching.per_token_admission.invalid]
- GIVEN evidence whose latency admission is reported as per-request wall latency across different capacity workloads
- WHEN the gate validates the admission basis
- THEN the gate MUST reject that basis
- AND the change MUST record the corrected per-token comparison without claiming the wall-latency basis was meaningful.

### Requirement: Fused RWKV-7 WKV kernel
r[tenstorrent.rwkv7_p150x2.fused_wkv_kernel] The runtime MAY provide a package-matched Blackhole kernel for token-at-a-time 64-wide RWKV-7 heads. The kernel MUST preserve the accepted WKV equation, FP32 recurrent state, BF16 output boundary, unique state ownership, and exact generated-token parity. The runtime MUST use the fused path only after capability validation and stable physical admission pass.

#### Scenario: Supported fused execution passes
r[tenstorrent.rwkv7_p150x2.fused_wkv_kernel.valid]
- GIVEN supported Blackhole hardware, 64-wide heads, BF16 model vectors, FP32 recurrent state, and unique state-tile ownership
- WHEN the fused WKV operator executes a decode step
- THEN its state and output MUST match the pure reference within profile-owned intermediate tolerances
- AND the complete workload MUST preserve exact FP32-reference generated tokens.

#### Scenario: Unsupported input fails closed
r[tenstorrent.rwkv7_p150x2.fused_wkv_kernel.invalid_input]
- GIVEN an unsupported architecture, shape, dtype, layout, alias, or state-ownership plan
- WHEN fused-path capability validation runs
- THEN validation MUST reject the fused dispatch before device execution
- AND the runtime MUST retain the admitted generic path.

#### Scenario: Hazard evidence is incomplete
r[tenstorrent.rwkv7_p150x2.fused_wkv_kernel.missing_audit]
- GIVEN a custom kernel without complete applicable LLK hazard, instruction-latency, performance, simulator, or parity evidence
- WHEN physical-run admission evaluates the candidate
- THEN admission MUST fail before silicon execution.

#### Scenario: Launches fall but throughput does not improve
r[tenstorrent.rwkv7_p150x2.fused_wkv_kernel.no_gain]
- GIVEN a parity-correct candidate with fewer launches but no stable physical throughput gain
- WHEN runtime selection evaluates the candidate
- THEN the runtime MUST keep the generic path
- AND the evidence package MUST retain the measured counterexample without claiming an improvement.

### Requirement: Bounded persistent device decode
r[tenstorrent.rwkv7_p150x2.persistent_device_decode] The fast runtime MAY provide a persistent device decode mode that carries selected token IDs and FP32 recurrent state across bounded decode steps. The mode MUST validate all binding identities before start, MUST provide bounded completion and cancellation, and MUST perform no per-token host submission inside its admitted persistent region. The runtime MUST retain normal dispatch as a fail-closed fallback.

#### Scenario: Persistent decode passes
r[tenstorrent.rwkv7_p150x2.persistent_device_decode.valid]
- GIVEN valid runtime, checkpoint, program, weight, token, state, slot, output, and profile identities
- AND a request within the profile-owned step, time, and output bounds
- WHEN persistent decode executes
- THEN the device MUST carry the selected token and FP32 recurrent state between steps
- AND the host MUST perform no per-token submission inside the persistent region
- AND generated tokens MUST exactly match normal dispatch and the pinned FP32 CPU reference.

#### Scenario: Binding identity is stale
r[tenstorrent.rwkv7_p150x2.persistent_device_decode.stale_binding]
- GIVEN a stale or mismatched token, state, slot, output, weight, program, runtime, checkpoint, or profile identity
- WHEN persistent capability validation runs
- THEN it MUST reject persistent execution before the device loop starts
- AND the runtime MUST retain normal dispatch.

#### Scenario: Cancellation or timeout occurs
r[tenstorrent.rwkv7_p150x2.persistent_device_decode.bounded_exit]
- GIVEN an active persistent request that receives cancellation or reaches a profile-owned bound
- WHEN the device and host process the bounded exit
- THEN they MUST reach one deterministic terminal state
- AND the slot MUST NOT return to the free list until completion acknowledgement and state zeroing finish.

#### Scenario: Host submissions remain in the token loop
r[tenstorrent.rwkv7_p150x2.persistent_device_decode.not_persistent]
- GIVEN a candidate that still performs one or more host submissions for each token inside the measured region
- WHEN persistent-mode admission evaluates the profiler evidence
- THEN admission MUST fail
- AND the runtime MUST NOT describe the candidate as persistent decode.

#### Scenario: Persistent path has no stable gain
r[tenstorrent.rwkv7_p150x2.persistent_device_decode.no_gain]
- GIVEN a parity-correct persistent candidate without stable physical throughput improvement
- WHEN runtime selection evaluates the candidate
- THEN the runtime MUST keep normal dispatch as the default
- AND the evidence package MUST retain the measured blocker without claiming improvement.

### Requirement: Profile-selected persistent decode windows
r[tenstorrent.rwkv7_p150x2.persistent_decode_windows] The runtime MAY admit multiple finite persistent decode windows. The base profile MUST own a closed, ordered set of window lengths and a maximum trace bound. The runtime MUST derive request-local lifecycle limits only from a listed window, MUST preserve the base profile identity, and MUST retain normal dispatch for every unsupported window.

#### Scenario: Listed longer window passes
r[tenstorrent.rwkv7_p150x2.persistent_decode_windows.valid]
- GIVEN a homogeneous request whose decode length is listed in the physically admitted profile
- AND exact token, FP32 recurrent-state, profiler, simulator, cleanup, and stable physical-gain evidence exists for that length
- WHEN capability validation selects persistent decode
- THEN one bounded trace replay MUST execute the complete selected window
- AND the host MUST perform no per-token model submission inside that window.

#### Scenario: Conversational prompt runs
r[tenstorrent.rwkv7_p150x2.persistent_decode_windows.real_prompt]
- GIVEN a tokenizer-backed conversational prompt with a pinned FP32 CPU-reference continuation
- WHEN an admitted persistent window executes the prompt
- THEN generated tokens MUST exactly match normal dispatch and the pinned continuation
- AND the final FP32 recurrent-state BLAKE3 MUST exactly match normal dispatch.

#### Scenario: Window is not listed
r[tenstorrent.rwkv7_p150x2.persistent_decode_windows.unlisted]
- GIVEN a positive bounded decode length that is not in the profile window set
- WHEN capability validation runs
- THEN validation MUST reject trace capture before device execution
- AND the runtime MUST select normal dispatch with an explicit fallback reason.

#### Scenario: Cancellation arrives before replay
r[tenstorrent.rwkv7_p150x2.persistent_decode_windows.cancelled]
- GIVEN an admitted request whose cancellation is visible before trace replay
- WHEN the runtime processes the cancellation
- THEN the trace MUST NOT execute
- AND the lifecycle MUST acknowledge cancellation, zero recurrent state, and release the slot in that order.

#### Scenario: Longer window exceeds a bound
r[tenstorrent.rwkv7_p150x2.persistent_decode_windows.bound_exceeded]
- GIVEN a listed window whose trace exceeds its time, output, or trace-capacity bound
- WHEN runtime validation or execution detects the excess
- THEN the window MUST fail closed without production admission
- AND the accepted shorter window MUST remain available.

#### Scenario: Longer window has no stable gain
r[tenstorrent.rwkv7_p150x2.persistent_decode_windows.no_gain]
- GIVEN a parity-correct longer window without stable one-card and two-card physical gain
- WHEN production admission evaluates its order-balanced evidence
- THEN the longer window MUST remain unadmitted
- AND the runtime MUST NOT generalize the shorter-window performance claim.

### Requirement: Persistent decode production telemetry
r[tenstorrent.rwkv7_p150x2.persistent_decode_telemetry] Every successful fast-worker receipt MUST contain one schema-versioned, aggregate-only persistent decode telemetry event. The event MUST record eligibility, selected dispatch mode, selected window, fallback reason, bounded timing values, terminal outcome counts, parity status, and cleanup status. The runtime MUST derive categorical values from a pure typed core. Evidence validation MUST reject telemetry that disagrees with the detailed worker receipt.

#### Scenario: Persistent execution emits telemetry
r[tenstorrent.rwkv7_p150x2.persistent_decode_telemetry.valid]
- GIVEN an eligible request that completes an admitted persistent window
- WHEN the fast worker writes its receipt
- THEN telemetry MUST report persistent mode, the selected window, no fallback, passed parity, and complete cleanup
- AND setup, replay, and cleanup durations MUST be finite and non-negative.

#### Scenario: Normal fallback emits telemetry
r[tenstorrent.rwkv7_p150x2.persistent_decode_telemetry.fallback]
- GIVEN a request that does not qualify for persistent dispatch
- WHEN the fast worker completes with normal dispatch
- THEN telemetry MUST report normal mode and the exact fallback reason
- AND trace timing, parity, cleanup, and terminal lifecycle fields MUST make no persistent-execution claim.

#### Scenario: Terminal event occurs before replay
r[tenstorrent.rwkv7_p150x2.persistent_decode_telemetry.terminal]
- GIVEN an admitted request with cancellation, timeout, or failure visible before replay
- WHEN the worker acknowledges the event and releases the slot
- THEN telemetry MUST report the exact terminal outcome count and complete cleanup
- AND telemetry MUST report zero trace replay time and no parity claim.

#### Scenario: Telemetry disagrees with evidence
r[tenstorrent.rwkv7_p150x2.persistent_decode_telemetry.invalid]
- GIVEN telemetry whose mode, window, fallback, parity, timing, terminal counts, or cleanup status disagrees with the detailed receipt
- WHEN the evidence guard validates the receipt
- THEN validation MUST fail with a typed telemetry error.

#### Scenario: Telemetry contains high-cardinality data
r[tenstorrent.rwkv7_p150x2.persistent_decode_telemetry.aggregate_only]
- GIVEN a telemetry event that contains prompt text, token IDs, request IDs, paths, binding identities, or unbounded labels
- WHEN the evidence guard validates the telemetry schema
- THEN validation MUST reject the event.

### Requirement: Persistent decode monitoring and alerts
r[tenstorrent.rwkv7_p150x2.persistent_decode_monitoring] The system MUST provide a deterministic bounded collector that validates persistent decode telemetry, emits aggregate production metrics, classifies fixed safety alerts, and recommends fail-closed persistent admission disablement for every critical result.

#### Scenario: Typed monitoring policy is valid
r[tenstorrent.rwkv7_p150x2.persistent_decode_monitoring.policy]
- GIVEN a strict policy with bounded event count and fallback thresholds
- WHEN Nickel exports the policy and Rust parses it
- THEN both layers accept the same policy values and fixed actions

#### Scenario: Valid receipt batch is collected
r[tenstorrent.rwkv7_p150x2.persistent_decode_monitoring.collect]
- GIVEN an explicit non-empty batch of valid worker receipts within the configured limit
- WHEN the monitoring core collects the batch
- THEN it emits bounded aggregates with ordered event BLAKE3 values and one deterministic batch BLAKE3
- AND it excludes paths, prompts, tokens, request identifiers, and binding identities

#### Scenario: Critical safety signal is observed
r[tenstorrent.rwkv7_p150x2.persistent_decode_monitoring.critical]
- GIVEN telemetry with parity failure, cleanup failure, timeout, terminal failure, unsupported persistent admission, malformed content, or duplicate event bytes
- WHEN the monitoring core classifies the batch
- THEN it emits the matching fixed critical alert
- AND it recommends persistent admission disablement

#### Scenario: Fallback rate exceeds policy
r[tenstorrent.rwkv7_p150x2.persistent_decode_monitoring.fallback]
- GIVEN enough accepted events to satisfy the configured minimum sample count
- AND the normal-dispatch fallback rate exceeds the configured maximum
- WHEN the monitoring core classifies the batch
- THEN it emits one fixed warning with the configured investigation action

#### Scenario: Operator runs the collector command
r[tenstorrent.rwkv7_p150x2.persistent_decode_monitoring.command]
- GIVEN a typed policy path and explicit worker receipt paths
- WHEN the operator runs the monitoring command
- THEN the command emits one monitoring receipt
- AND it returns a stable status for clean, warning, or critical results

#### Scenario: Monitoring input is invalid or unbounded
r[tenstorrent.rwkv7_p150x2.persistent_decode_monitoring.invalid]
- GIVEN an empty batch, oversized batch, invalid policy, non-finite threshold, unknown field, or unsupported schema
- WHEN the monitoring core validates the input
- THEN collection fails before it emits an admission recommendation

### Requirement: Bounded persistent decode canary
r[tenstorrent.rwkv7_p150x2.persistent_decode_canary] The system MUST provide a bounded physical canary controller whose BLAKE3-bound runtime interlock forces normal dispatch after every held, disabled, malformed, mismatched, or critical canary state.

#### Scenario: Canary policy is valid
r[tenstorrent.rwkv7_p150x2.persistent_decode_canary.policy]
- GIVEN the strict production canary policy and matched throughput profile
- WHEN Nickel exports them and Rust validates their cross-field budgets
- THEN machine, device, request, duration, traffic, scheduling, profile, window, and gain limits agree

#### Scenario: Armed interlock permits candidate execution
r[tenstorrent.rwkv7_p150x2.persistent_decode_canary.interlock]
- GIVEN an armed interlock with matching file BLAKE3, canary binding, profile identity, and bounded limits
- WHEN an eligible canary worker selects dispatch
- THEN persistent trace remains available for the candidate worker

#### Scenario: Runtime receives a blocked interlock
r[tenstorrent.rwkv7_p150x2.persistent_decode_canary.runtime]
- GIVEN a held, disabled, malformed, hash-mismatched, binding-mismatched, or profile-mismatched interlock
- WHEN any worker requests persistent dispatch, including a candidate override
- THEN the runtime uses normal dispatch with `canary_interlock_blocked`

#### Scenario: Canary evidence is evaluated
r[tenstorrent.rwkv7_p150x2.persistent_decode_canary.evaluate]
- GIVEN a bounded completed benchmark, strict candidate telemetry, and matched physical A/B evidence
- WHEN the pure canary core evaluates parity, cleanup, terminal outcomes, gain, duration, and request budgets
- THEN it emits a deterministic final interlock and canary receipt

#### Scenario: Operator runs the canary command
r[tenstorrent.rwkv7_p150x2.persistent_decode_canary.command]
- GIVEN explicit policy, throughput, workload, monitoring, and new output-root paths
- WHEN the operator runs the packaged canary command
- THEN it arms the interlock, runs one matched benchmark under the policy timeout, and atomically writes the final state
- AND one-card samples balance concurrent workers across both cards while dual-card samples alternate order

#### Scenario: Canary evidence is invalid or unsafe
r[tenstorrent.rwkv7_p150x2.persistent_decode_canary.invalid]
- GIVEN a policy mismatch, request overflow, timeout, benchmark failure, invalid A/B receipt, warning, critical alert, parity failure, cleanup failure, or terminal failure
- WHEN the controller evaluates the run
- THEN it does not emit `retain`
- AND every critical or invalid result emits `disable`

#### Scenario: Physical canary passes
r[tenstorrent.rwkv7_p150x2.persistent_decode_canary.physical]
- GIVEN one bounded run on physical devices `0` and `1`
- WHEN matched normal and persistent samples complete with clean candidate monitoring
- THEN the retained canary receipt proves exact parity, complete cleanup, stable positive gain, bounded duration, and final `retain`
