# Qwen38 P150x2 Runtime Specification

## Purpose

Defines the `qwen38-p150x2-runtime` capability.

## Requirements

### Requirement: Decode trace replay is bounded and fail-closed
r[tenstorrent.qwen38_p150x2.traced_decode] The service MUST capture only an eligible static decode step. It MUST bind the trace to immutable runtime, model, slot, buffer, shape, dtype, and generation identities. It MUST restore recurrent state after capture. It MUST retain eager decode for every unsupported or invalid trace case.

#### Scenario: Eligible decode uses trace replay
r[tenstorrent.qwen38_p150x2.traced_decode.valid]
- GIVEN loaded weights, warmed programs, stable request buffers, and restored recurrent state
- WHEN an eligible greedy decode step executes
- THEN the service MUST refresh all mutable trace inputs before replay
- AND the replay MUST consume the prior request state exactly once.

#### Scenario: Invalid trace binding falls back
r[tenstorrent.qwen38_p150x2.traced_decode.fallback]
- GIVEN a missing, stale, mismatched, oversized, or unsupported trace binding
- WHEN decode admission validates the binding
- THEN the service MUST select eager decode before token execution
- AND the receipt MUST name the bounded fallback reason.

#### Scenario: Trace capture changes recurrent state
r[tenstorrent.qwen38_p150x2.traced_decode.state_drift]
- GIVEN trace capture output whose recurrent state differs from the pre-capture snapshot
- WHEN post-capture validation runs before the first request
- THEN validation MUST restore the snapshot and verify its identity
- AND startup MUST fail if restoration cannot be proved.

### Requirement: Decode buffers have explicit request ownership
r[tenstorrent.qwen38_p150x2.persistent_decode_buffers] The service MUST allocate a bounded set of stable decode slots. One admitted request MUST own exactly one slot. Each slot MUST own its token, position, RoPE, page-table, KV, GDN, output, and trace-binding resources. Slot reuse MUST follow a complete reset. Cleanup failure MUST quarantine the slot.

#### Scenario: Stable buffers serve one request
r[tenstorrent.qwen38_p150x2.persistent_decode_buffers.valid]
- GIVEN an empty slot and one admitted request
- WHEN the request executes two or more decode steps
- THEN every step MUST use the same slot generation and stable device resources
- AND no other live request MAY use those resources.

#### Scenario: Slot cleanup fails
r[tenstorrent.qwen38_p150x2.persistent_decode_buffers.cleanup_failure]
- GIVEN a completed, cancelled, timed-out, or failed request whose slot cannot pass reset validation
- WHEN the service releases request ownership
- THEN the slot MUST enter quarantine
- AND the service MUST reject reuse until a complete reset succeeds.

#### Scenario: Invalid slot transition is requested
r[tenstorrent.qwen38_p150x2.persistent_decode_buffers.invalid_transition]
- GIVEN a slot transition outside the typed lifecycle
- WHEN the pure lifecycle core validates the transition
- THEN validation MUST fail before device or filesystem effects occur.

### Requirement: Greedy selection transfers bounded shard winners
r[tenstorrent.qwen38_p150x2.shard_local_argmax] Each TP2 vocabulary shard MUST produce one finite maximum and one global token index. The host MUST select only from validated shard winners. Ties MUST select the lowest global token index. Admitted traced greedy decode MUST NOT transfer the complete vocabulary logits to the host.

#### Scenario: Valid shard winners select one token
r[tenstorrent.qwen38_p150x2.shard_local_argmax.valid]
- GIVEN complete non-overlapping shard ranges and one finite winner from each shard
- WHEN the pure reducer selects the global winner
- THEN it MUST return the token with the greatest logit
- AND equal logits MUST select the lowest global token index.

#### Scenario: Malformed shard result is rejected
r[tenstorrent.qwen38_p150x2.shard_local_argmax.invalid]
- GIVEN a missing winner, duplicate range, overlapping range, out-of-range index, or non-finite maximum
- WHEN shard-result validation runs
- THEN validation MUST fail before generated text is published.

### Requirement: Decode speedup claims require matched physical evidence
r[tenstorrent.qwen38_p150x2.decode_performance_admission] A traced candidate MUST match the eager path's greedy generated tokens for the profile corpus. A speedup claim MUST use order-balanced physical A/B evidence. The evidence MUST bind both paths to the same closure, checkpoint, tokenizer, workload, mesh, and generation settings. Nickel MUST own every admission bound.

#### Scenario: Physical candidate passes
r[tenstorrent.qwen38_p150x2.decode_performance_admission.valid]
- GIVEN complete physical A/B evidence with exact token parity and clean slot reset
- WHEN the pure admission core evaluates decode rate, TTFT, and tail latency
- THEN every profile-owned bound MUST pass
- AND the receipt MAY admit traced decode for the bound closure and workload.

#### Scenario: Candidate lacks complete evidence
r[tenstorrent.qwen38_p150x2.decode_performance_admission.invalid]
- GIVEN missing parity, cleanup, machine, attribution, latency, throughput, or identity evidence
- WHEN admission evaluates the candidate
- THEN it MUST withhold the speedup claim
- AND it MUST retain eager decode as the admitted fallback.

### Requirement: Legacy continuous batching remains unsupported without separate admission

r[tenstorrent.qwen38_p150x2.continuous_batching_non_support]

The repository MUST NOT treat the bounded active-prefix batch path as proof of a general continuous scheduler. Arbitrary slot recycling, mixed generation bounds, non-prefix occupancy, live prefill during decode, and dynamic batch compaction remain unsupported until separate typed changes admit them. The older unadmitted single-stream trace receipt MUST NOT override the later bounded batching requirements. A batch-specific receipt MAY admit only the exact width, workload, state layout, and identity that it evaluates.

#### Scenario: Bounded active-prefix batching is evaluated

- GIVEN the accepted isolated batch path and its batch-specific typed profile
- WHEN exact B=1 versus B=4 physical roots are evaluated
- THEN the receipt MAY decide only bounded B=4 throughput admission
- AND it MUST NOT claim general continuous batching.

#### Scenario: An operator requests unsupported scheduling

- GIVEN mixed generation bounds, non-prefix occupancy, slot recycling, live prefill during decode, or dynamic compaction
- WHEN request execution evaluates the unsupported schedule
- THEN the service MUST reject or defer that schedule before state mutation
- AND the bounded B=4 receipt MUST NOT authorize it.

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

### Requirement: Traced decode uses request-bounded prefill capacity
r[tenstorrent.qwen38_p150x2.traced_decode_ttft] The traced-decode candidate MUST size each prefill page table from the bounded prompt and generation token counts. The prefill table MUST be a monotonic prefix of the stable decode table. The trace-bound decode table MUST retain its maximum profile capacity, shape, contents, and device address across requests. Invalid or capacity-exceeding requests MUST fail before prefill dispatch.

#### Scenario: Request uses a bounded prefill table
r[tenstorrent.qwen38_p150x2.traced_decode_ttft.valid]
- GIVEN positive prompt and generation counts within the trace capacity
- WHEN the pure block-count rule and adapter build the prefill table
- THEN its width MUST equal the rounded-up required block count
- AND its entries MUST equal the same prefix of the stable decode table.

#### Scenario: Request exceeds trace capacity
r[tenstorrent.qwen38_p150x2.traced_decode_ttft.capacity]
- GIVEN a prompt and generation bound that need more blocks than the trace capacity
- WHEN request block-count validation runs
- THEN it MUST reject the request before prefill dispatch
- AND it MUST NOT resize or replace the trace-bound decode table.

#### Scenario: Prefill width changes trace binding
r[tenstorrent.qwen38_p150x2.traced_decode_ttft.binding]
- GIVEN a request-bounded prefill table and an existing decode trace
- WHEN trace binding validation runs before replay
- THEN the stable decode page-table identity MUST remain unchanged
- AND any identity drift MUST select the fail-closed eager path.

### Requirement: TTFT admission uses phase-attributed matched evidence
r[tenstorrent.qwen38_p150x2.traced_decode_ttft_evidence] Each measured eager and candidate response MUST record finite non-negative reset, prefill-device, first-token-readback, trace-preparation, and total TTFT durations. It MUST record prefill and decode page-table widths. The total TTFT boundaries MUST remain unchanged. Admission MUST use order-balanced matched physical evidence and MUST reject missing, non-finite, inconsistent, or identity-mismatched observations.

#### Scenario: Phase evidence is complete
r[tenstorrent.qwen38_p150x2.traced_decode_ttft_evidence.valid]
- GIVEN matched physical responses with complete phase timings and table widths
- WHEN the deterministic receipt collector evaluates them
- THEN it MUST retain phase medians and total TTFT ratios
- AND admission MAY pass only when every existing correctness and performance gate passes.

#### Scenario: Phase evidence is incomplete
r[tenstorrent.qwen38_p150x2.traced_decode_ttft_evidence.invalid]
- GIVEN a missing phase, non-finite duration, negative duration, invalid width, or mismatched identity
- WHEN receipt validation runs
- THEN it MUST reject the candidate evidence
- AND production MUST remain eager.

#### Scenario: Physical candidate passes
r[tenstorrent.qwen38_p150x2.traced_decode_ttft_evidence.physical]
- GIVEN matched `eager, candidate, candidate, eager` physical roots on device IDs `0` and `1`
- WHEN exact parity, reset, decode, TTFT, tail, phase, and identity gates pass
- THEN traced decode MAY become physically admitted
- AND the admitted profile MUST bind the new receipt BLAKE3.

### Requirement: Prefill profiling validates device program records
r[tenstorrent.qwen38_p150x2.prefill_program_profile] The service MAY enable a bounded prefill diagnostic that uses package-matched TT-Metal real-time program records. The deterministic core MUST validate every record before it computes a summary. It MUST aggregate exact kernel-source signatures by physical chip without importing TTNN types. The normal eager and candidate paths MUST have no profiling callback, drain, or response payload unless the operator enables the diagnostic.

#### Scenario: Valid program records are summarized
r[tenstorrent.qwen38_p150x2.prefill_program_profile.valid]
- GIVEN positive device durations, admitted chip IDs, positive frequencies, and non-empty kernel-source paths
- WHEN the pure summary core evaluates the prefill records
- THEN it MUST report deterministic per-signature and per-chip counts and durations
- AND it MUST sort equal totals by the exact kernel-source signature.

#### Scenario: A program record is malformed
r[tenstorrent.qwen38_p150x2.prefill_program_profile.invalid]
- GIVEN an invalid chip, runtime ID, timestamp interval, frequency, or kernel-source path
- WHEN summary validation runs
- THEN it MUST reject the complete profile before attribution
- AND it MUST NOT publish an optimization claim.

#### Scenario: Profiling is not enabled
r[tenstorrent.qwen38_p150x2.prefill_program_profile.disabled]
- GIVEN the production eager service or an unprofiled candidate service
- WHEN prefill executes
- THEN the service MUST NOT register a profiling callback or wait for receiver drain
- AND its existing TTFT boundaries and production behavior MUST remain unchanged.

### Requirement: Prefill attribution uses matched diagnostic evidence
r[tenstorrent.qwen38_p150x2.prefill_profile_evidence] A prefill attribution claim MUST use physical P150x2 profile roots in the exact order `eager,candidate,candidate,eager`. Each response MUST bind the runtime, package, model, prompt, table width, and diagnostic measurement mode. Profiled responses MUST be ineligible for TTFT admission. An optimization MAY be selected only when duplicate roots have complete signature coverage and attribute a material eager-versus-candidate device-program difference.

#### Scenario: Matched profile roots identify a difference
r[tenstorrent.qwen38_p150x2.prefill_profile_evidence.valid]
- GIVEN complete balanced physical roots with matching identities and profile coverage
- WHEN the deterministic comparator evaluates signature totals by chip
- THEN it MUST report the strongest checked eager-versus-candidate difference
- AND it MAY name one source-backed optimization candidate without admitting it.

#### Scenario: Profile evidence is incomplete
r[tenstorrent.qwen38_p150x2.prefill_profile_evidence.invalid]
- GIVEN an inactive profiler, missing record, malformed summary, wrong execution order, identity drift, or unmatched corpus
- WHEN attribution validation runs
- THEN it MUST reject the attribution result
- AND production MUST remain eager.

#### Scenario: Profiled responses reach TTFT admission
r[tenstorrent.qwen38_p150x2.prefill_profile_evidence.non_admission]
- GIVEN a response marked as diagnostic prefill profiling
- WHEN the TTFT admission receipt evaluates it
- THEN admission MUST reject the response regardless of its measured TTFT
- AND a later unprofiled matched physical round MUST remain required.

### Requirement: Host timeline diagnostics use package-matched tooling and stay diagnostic

r[tenstorrent.qwen38_p150x2.host_timeline_profile] The repository MAY provide host-side Tracy CLI tooling built from the package-matched TT-Metal vendored fork so the packaged Python `tracy` module can capture host zones. Host timeline capture MUST be an operator-enabled diagnostic that runs only under authorized device isolation. Host timeline observations MUST NOT satisfy any admission gate. A host attribution claim MUST come from a deterministic comparator over validated captures from matched balanced physical roots.

#### Scenario: Matched captures expose host spans

r[tenstorrent.qwen38_p150x2.host_timeline_profile.valid]

- GIVEN validated captures from matched `eager,candidate,candidate,eager` physical roots with consistent runtime, package, model, and mesh identities
- WHEN the deterministic comparator evaluates exported host zones
- THEN it MUST report ranked inter-enqueue gaps and phase totals per decode mode
- AND it MAY name one host-side attribution result without admitting any optimization.

#### Scenario: Host timeline evidence is invalid

r[tenstorrent.qwen38_p150x2.host_timeline_profile.invalid]

- GIVEN a missing capture, malformed zone record, identity drift, or wrong execution order
- WHEN comparator validation runs
- THEN it MUST reject the attribution result
- AND production MUST remain unchanged.

#### Scenario: Host profiling is not enabled

r[tenstorrent.qwen38_p150x2.host_timeline_profile.disabled]

- GIVEN the production eager service or an unprofiled candidate service
- WHEN requests execute
- THEN the service MUST NOT run a capture tool or change its response payloads
- AND existing TTFT boundaries and admission behavior MUST remain unchanged.

### Requirement: Host timeline boundaries avoid Python call profiling

r[tenstorrent.qwen38_p150x2.host_timeline_signposts] Host timeline diagnostics MUST use bounded Tracy signpost messages without enabling Python call profiling. The comparator MUST validate one ordered start/end pair and MUST retain duplicate stability gates. Results MUST remain diagnostic-only.

#### Scenario: Signpost capture is stable

- GIVEN matched balanced physical roots with valid signposts and C++ enqueue zones
- WHEN duplicate validation passes
- THEN the comparator MAY report host attribution
- AND it MUST NOT admit an optimization.

#### Scenario: Signposts or duplicates are invalid

- GIVEN missing, misordered, or repeated signposts, or unstable duplicate roots
- WHEN validation runs
- THEN the comparator MUST reject attribution
- AND production MUST remain eager.

### Requirement: Host timeline roots use bounded repeated sampling

r[tenstorrent.qwen38_p150x2.host_timeline_sampling] Each host-timeline root MUST contain at least five complete ordered prefill signpost intervals after warm-up. The comparator MUST calculate each interval independently, use the median interval metrics as the root summary, reject malformed signpost pairs, and retain the `1.25` duplicate-root bound. Physical execution MUST retain two roots per mode in exact `eager,candidate,candidate,eager` order. Results MUST remain diagnostic-only.

#### Scenario: Repeated roots are stable

- GIVEN five valid samples in each immutable physical root
- WHEN both duplicate comparisons pass
- THEN the receipt MAY report diagnostic host attribution
- AND it MUST NOT admit an optimization.

#### Scenario: Sampling evidence is invalid

- GIVEN too few samples, malformed pairs, identity drift, or an unstable duplicate metric
- WHEN validation runs
- THEN the receipt MUST reject attribution
- AND production MUST remain eager.

### Requirement: MTP target verification requires bounded hybrid-state evidence

r[tenstorrent.qwen38_p150x2.mtp_verification_probe] The repository MAY provide an operator-enabled teacher-forced target-verification probe for the pinned P150x2 runtime. The probe MUST remain diagnostic-only. It MUST compare widths 1, 2, and 4 with matched sequential traced decode after the same prompt state. It MUST use at least five complete samples per width. Every sample MUST preserve target-token parity and bounded GDN recurrent and convolution state equivalence. The deterministic evaluator MUST reject missing identities, unstable samples, invalid durations, wrong widths, token mismatch, or state mismatch. A valid target-only result MUST NOT select or admit MTP without a measured drafter, local acceptance, rollback correctness, and matched end-to-end evidence.

#### Scenario: Teacher-forced target verification is complete

- GIVEN matched P150x2 sequential and teacher-forced samples for widths 1, 2, and 4
- WHEN all identity, timing, target-token, and hybrid-state checks pass
- THEN the receipt MAY report target-verification amortization for the pinned runtime
- AND it MUST state that MTP remains unselected and unadmitted.

#### Scenario: Verification evidence is incomplete

- GIVEN a missing width, too few samples, identity drift, invalid duration, target-token mismatch, or GDN state mismatch
- WHEN the deterministic evaluator runs
- THEN it MUST reject target-verification feasibility
- AND production MUST remain eager.

### Requirement: Native short-width target verification preserves hybrid state

r[tenstorrent.qwen38_p150x2.native_mtp_verifier] The repository MAY provide an operator-enabled native target verifier for the pinned P150x2 runtime. The verifier MUST remain diagnostic-only. It MUST process widths 1, 2, and 4 layer by layer. It MUST batch weight-bearing projections, MLPs, and target-logit work across positions. It MUST preserve token-ordered high-precision GDN recurrence and convolution updates. It MUST serialize paged-KV writes when candidate rows alias one physical block. It MUST execute the complete width-specific hybrid forward through one captured trace replay. It MUST return target logits for every position. Physical evidence MUST use at least five complete samples per width and compare exact target tokens, bounded full logits, final GDN recurrent and convolution state, written full-attention KV positions, identities, durations, and replay stability against sequential traced decode. Full-logit host readback MUST remain separately visible and MUST NOT stand in for device-resident acceptance timing. A missing, mismatched, unstable, or slower verifier MUST select no optimization. A valid target-only result MUST NOT select or admit MTP without drafter timing, local acceptance, rejection rollback, end-to-end correctness, and admission evidence.

#### Scenario: Native target verification is valid

- **GIVEN** the exact pinned P150x2 runtime, package, source, checkpoint, tokenizer, mesh, policy, prompt state, and physical device identities
- **WHEN** the operator runs five complete native-verifier samples for each required width
- **THEN** every target token, full-logit metric, GDN state metric, paged-KV metric, duration, identity, and stability check satisfies policy, while MTP remains unselected and unadmitted

#### Scenario: Native target verification fails closed

- **GIVEN** a verifier that repeats full-model steps, loses prompt state, races paged-KV writes, omits a position logit, changes a target token, exceeds a state bound, has unstable timing, or fails to amortize target work
- **WHEN** the deterministic evaluator checks its evidence
- **THEN** it rejects the verifier, selects no optimization, and preserves eager production

### Requirement: MTP drafter evidence binds shifted inputs, acceptance, and rollback

r[tenstorrent.qwen38_p150x2.mtp_drafter_probe] The repository MAY provide an operator-enabled Qwen3.8 MTP drafter probe for the pinned P150x2 runtime. The probe MUST remain diagnostic-only. It MUST bind the exact checkpoint MTP tensors and their shapes. It MUST share the target embedding and LM head. It MUST separately normalize shifted next-token embeddings and post-final-norm target hidden states before the MTP FC projection. It MUST run one full-attention MTP decoder layer with its own position-ordered KV cache. It MUST generate exactly three greedy drafts. Physical evidence MUST compare those drafts with aligned target logits across at least five fixed prompts. It MUST record accepted-prefix length, draft time, target-verifier time, emitted-token economics, identities, stability, and cleanup. MTP MUST remain unselected when weight loading, parity, acceptance, rollback, target correctness, or economics is missing or invalid. Target GDN state MUST restore to the accepted prefix before any end-to-end MTP selection.

#### Scenario: Drafter evidence is complete

- **GIVEN** the exact pinned P150x2 package, runtime, checkpoint, tokenizer, MTP tensors, prompts, and physical device identities
- **WHEN** the operator runs the bounded three-draft probe
- **THEN** shifted-input alignment, local acceptance, timing, target correctness, rollback, stability, and non-admission are explicit in the receipt

#### Scenario: Drafter evidence fails closed

- **GIVEN** missing or drifted MTP weights, an uninitialized drafter cache, misaligned logits, unstable timing, insufficient acceptance, incorrect rollback, or slower emitted-token economics
- **WHEN** the deterministic evaluator checks the evidence
- **THEN** it rejects MTP selection and preserves eager production

### Requirement: Checkpoint layer types own hybrid dispatch

r[tenstorrent.qwen38_p150x2.layer_type_authority] The Qwen3.8 runtime MUST derive every layer's mixer type from the pinned checkpoint `layer_types` list. The list MUST contain exactly one recognized type for every declared text layer. A static mapper list MUST NOT override checkpoint dispatch. A legacy fallback MAY apply only when its length equals the declared layer count. Invalid or incomplete layer-type authority MUST fail before layer construction or weight loading.

#### Scenario: Pinned Qwen3.8 layer types are complete

- **GIVEN** the pinned 64-layer Qwen3.8 checkpoint with its complete `layer_types` list
- **WHEN** the pure resolver validates hybrid dispatch
- **THEN** it returns 48 linear-attention layers and 16 full-attention layers
- **AND** full-attention layers occur at every fourth index, starting at index 3.

#### Scenario: Layer-type authority is invalid

- **GIVEN** a missing non-legacy list, a list with the wrong length, or an unknown layer type
- **WHEN** the pure resolver validates hybrid dispatch
- **THEN** it rejects the configuration before layer construction or weight loading
- **AND** it does not substitute the obsolete eight-entry mapper set.

### Requirement: Decode component attribution uses matched trace replay

r[tenstorrent.qwen38_p150x2.decode_component_attribution] The repository MAY provide an operator-enabled component attribution probe for the pinned Qwen3.8 P150x2 runtime. The probe MUST remain diagnostic-only. It MUST measure repeated trace replay for embedding, representative GDN mixers, representative MLPs, representative full-attention mixers, and LM-head reduction. It MUST also measure synchronized host input update, bounded winner readback, and the complete traced decode step. Each retained group MUST contain at least five complete samples. The deterministic evaluator MUST reject missing identities, wrong layer types, invalid durations, unstable samples, or an additive component model outside its configured closure bounds. A rejected result MUST select no optimization and MUST preserve eager production.

#### Scenario: Component attribution is valid

- **GIVEN** the exact pinned package, runtime, checkpoint, topology, profile, prompt state, and representative layer set
- **WHEN** every component and the full decode path complete repeated trace replay within the stability and closure bounds
- **THEN** the receipt MUST report direct medians, weighted component shares, host costs, and the selected bottleneck
- **AND** the result MUST remain diagnostic and ineligible for production admission.

#### Scenario: Component attribution is invalid

- **GIVEN** a missing group, wrong layer type, identity mismatch, malformed duration, unstable sample set, or additive model outside its closure bounds
- **WHEN** the deterministic evaluator processes the evidence
- **THEN** it MUST reject attribution and name each bounded reason
- **AND** it MUST select no kernel, precision, collective, or production change.

### Requirement: Traced TTFT control separates cache capacity and parked-trace effects

r[tenstorrent.qwen38_p150x2.traced_ttft_control] The repository MAY provide an operator-enabled same-process TTFT control for the pinned Qwen3.8 P150x2 runtime. The control MUST remain diagnostic-only. It MUST use one loaded model, one bounded prompt, stable request-sized and trace-capacity KV cache sets, and balanced root order. It MUST compare cache capacity without a parked trace and compare trace-capacity prefill before, during, and after a parked decode trace. Each root MUST contain at least five complete retained samples after warm-up. The deterministic evaluator MUST reject missing identities, wrong route order, token mismatch, invalid cache widths, malformed durations, unstable roots, unstable duplicate roots, or incomplete cleanup. The lifecycle MUST stop if eager-service restoration fails. A rejected result MUST select no optimization and MUST preserve eager production.

#### Scenario: Same-process TTFT control is valid

r[tenstorrent.qwen38_p150x2.traced_ttft_control.valid]
- **GIVEN** the exact pinned package, runtime, checkpoint, prompt, topology, cache sets, route order, and physical device identities
- **WHEN** every root completes repeated TTFT samples within the configured stability bounds
- **THEN** the receipt MUST report root medians, cache-capacity ratios, parked-trace ratios, combined ratios, and any selected bounded mechanism
- **AND** the result MUST remain diagnostic and ineligible for production admission.

#### Scenario: Same-process TTFT control is invalid

r[tenstorrent.qwen38_p150x2.traced_ttft_control.invalid]
- **GIVEN** identity drift, wrong route order, token mismatch, invalid widths, malformed timing, unstable roots, incomplete cleanup, or failed service restoration
- **WHEN** the deterministic evaluator and lifecycle gates process the evidence
- **THEN** they MUST reject the control and name each bounded reason
- **AND** it MUST select no optimization or production change.

### Requirement: MTP draft selection uses bounded causal evidence

r[tenstorrent.qwen38_p150x2.mtp_draft_selection] The repository MAY provide a diagnostic P150x2 control that compares full-logit MTP draft selection with shard-local top-one selection. The control MUST use the exact pinned package identities, physical devices `0` and `1`, batch one, exactly three drafts, and at least five fixed prompts. It MUST warm both routes and execute exact per-prompt order `full,bounded,bounded,full`. The bounded route MUST read only one local index and value per vocabulary shard and MUST use the same deterministic cross-shard tie policy as traced target decode. The evaluator MUST reject missing roots, route drift, invalid durations, duplicate-root instability, draft-token mismatch, target-token mismatch, acceptance mismatch, state mismatch, rollback failure, insufficient selector gain, or emitted-token aggregate speedup not strictly greater than `1.0`. A valid result MAY select bounded MTP draft selection for further diagnostic work. It MUST NOT admit MTP, traced decode, reduced precision, or production behavior.

#### Scenario: Bounded draft selection is selected

- **GIVEN** exact matched full-readback and bounded-selector roots for all fixed prompts
- **WHEN** duplicate roots are stable, all tokens and states match, bounded selection is materially faster, and aggregate economics exceed break-even
- **THEN** the evaluator SHALL select bounded draft selection for the next diagnostic MTP stage while production remains eager

#### Scenario: Selector evidence fails closed

- **GIVEN** missing roots, route drift, unstable samples, invalid timing, any output mismatch, rollback failure, insufficient gain, or aggregate slowdown
- **WHEN** the deterministic evaluator runs
- **THEN** it MUST reject selector optimization and MUST select no production change

### Requirement: MTP draft tracing uses stable causal evidence

r[tenstorrent.qwen38_p150x2.mtp_draft_trace] The repository MAY provide a diagnostic P150x2 control that compares eager and persistent-trace execution of the fixed-shape MTP draft decode step. The control MUST use exact pinned identities, physical devices `0` and `1`, batch one, exactly three drafts, the bounded shard-local selector, and at least five fixed prompts. It MUST warm both routes and retain at least five complete samples per route and prompt in balanced `eager,traced,traced,eager` blocks. Trace capture time MUST be recorded and excluded from replay economics. The traced route MUST bind stable token, position, RoPE, feedback, MTP KV-cache, selector-output, and trace-output addresses. It MUST restore capture side effects before measurement or prove that repeated same-input writes at the captured position are exact. The evaluator MUST reject missing samples, unknown fields, route drift, invalid durations, address drift, duplicate-root spread above `1.25`, token mismatch, target mismatch, acceptance mismatch, MTP KV-prefix mismatch, target rollback failure, traced/eager draft ratio above the configured bound, or emitted-token aggregate speedup not strictly greater than `1.0`. A valid result MAY select traced MTP drafting for later adaptive-width investigation. It MUST NOT admit MTP, traced target decode, reduced precision, or production behavior.

#### Scenario: Traced drafting is selected

- **GIVEN** matched eager and traced roots for all fixed prompts
- **WHEN** every binding and output matches, retained roots are stable, traced drafting is materially faster, and aggregate economics exceed break-even
- **THEN** the evaluator SHALL select traced MTP drafting for the next diagnostic stage while production remains eager

#### Scenario: Trace evidence fails closed

- **GIVEN** missing evidence, route drift, address drift, unstable timing, invalid timing, any output or state mismatch, insufficient draft gain, or aggregate slowdown
- **WHEN** the deterministic evaluator runs
- **THEN** it MUST reject traced drafting and MUST select no production change

### Requirement: MTP rejection uses bounded device-resident rollback evidence

r[tenstorrent.qwen38_p150x2.mtp_kv_rollback] The repository MAY provide a diagnostic P150x2 control that compares MTP cache re-prefill with device-resident rollback. The control MUST use pinned identities, physical devices `0` and `1`, batch one, exactly three drafts, the bounded shard-local selector, and at least five fixed prompts. It MUST retain at least five complete samples per route and prompt in balanced `reprefill,snapshot,snapshot,reprefill` blocks. Before draft trace capture, the snapshot route MUST preserve only the MTP K/V rows that fixed-shape draft decode can overwrite and MUST preserve prefix feedback in preallocated device buffers. After target verification, it MUST restore every rejected row, retain every accepted row, process any accepted draft not already represented in feedback, and advance exactly one committed target token. It MUST compare the next draft token, feedback tensor, and committed MTP K/V prefix with re-prefill. The evaluator MUST reject missing or unknown evidence, route drift, invalid duration, identity drift, state mismatch, acceptance mismatch, duplicate-root recovery spread above `1.25`, or snapshot/re-prefill recovery ratio above the configured bound. A valid result MAY select snapshot rollback for later diagnostic MTP work. It MUST NOT admit MTP, traced target decode, reduced precision, or production behavior.

#### Scenario: Snapshot rollback is selected

- **GIVEN** matched re-prefill and snapshot roots for all fixed prompts
- **WHEN** next-round MTP state matches exactly, recovery roots are stable, and snapshot recovery is materially faster
- **THEN** the evaluator SHALL select snapshot rollback for later diagnostic MTP work while production remains eager

#### Scenario: Rollback evidence fails closed

- **GIVEN** missing evidence, route drift, unstable timing, invalid timing, state mismatch, or insufficient recovery gain
- **WHEN** the deterministic evaluator runs
- **THEN** it MUST reject snapshot rollback and MUST select no production change

### Requirement: MTP recovery stability uses one persistent trace

r[tenstorrent.qwen38_p150x2.mtp_recovery_stability]

The diagnostic P150x2 control MUST use one fixed-shape MTP trace and stable buffers for draft and recovery replay in one loaded process. Selection MUST require one trace capture, exact binding reuse, exact next-round state, complete cleanup, six roots per route and prompt, recovery spread at most `1.25`, and a snapshot/re-prefill median ratio at most `0.8`. The control MUST remain diagnostic-only and MUST NOT change eager production behavior.

#### Scenario: Persistent recovery trace passes

GIVEN a matched P150x2 package, runtime, checkpoint, mesh, device set, prompt set, route schedule, and stable trace binding
WHEN one loaded process captures one MTP trace during discarded warmup and reuses it for all measured draft and recovery steps
THEN the receipt records one capture, exact binding reuse, exact next-round state, complete cleanup, stable route groups, and the bounded recovery ratio

#### Scenario: Persistent recovery trace fails closed

GIVEN a missing or repeated capture, changed binding, state mismatch, incomplete cleanup, unstable route group, or excessive recovery ratio
WHEN the deterministic evaluator reads the observations
THEN it rejects persistent recovery tracing and preserves eager production behavior

### Requirement: Staged MTP recovery replay preserves the last completed boundary

r[tenstorrent.qwen38_p150x2.mtp_recovery_stage_probe]

The diagnostic P150x2 command MUST run one configured prompt and recovery route through the unchanged recovery path. It MUST atomically preserve ordered progress before and after trace, replay, recovery, verification, and cleanup boundaries. Unknown, repeated, time-regressing, or route-incompatible stages MUST fail closed. The staged result MUST remain admission-ineligible and MUST NOT change eager production behavior.

#### Scenario: Staged replay completes

GIVEN a matched P150x2 package, runtime, checkpoint, mesh, prompt, and recovery route
WHEN the staged command completes within its named bound
THEN the receipt and progress ledger identify every completed boundary and preserve exact diagnostic correctness fields

#### Scenario: Staged replay reaches its bound

GIVEN a matched staged command that does not complete
WHEN the external watchdog terminates the command
THEN the atomic progress ledger preserves the last completed boundary and eager production health is restored

### Requirement: Same-process target prewarm isolates post-recovery compilation

r[tenstorrent.qwen38_p150x2.mtp_target_validation_prewarm]

The diagnostic P150x2 command MUST prewarm the same prompt through the existing nonpersistent target-validation path before persistent MTP recovery. It MUST preserve ordered prewarm and measured boundaries with nonnegative program-cache observations. It MUST reject a prewarm whose target-validation signature differs from the measured signature. The prewarm and its timing MUST remain admission-ineligible. The command MUST NOT change eager production behavior.

#### Scenario: Exact prewarm completes before the persistent route

GIVEN the pinned P150x2 package, runtime, checkpoint, prompt, and recovery route
WHEN the operator runs the target-prewarm diagnostic command
THEN the command preserves prewarm progress, runs persistent recovery in the same process, and compares exact target-validation signatures before receipt creation

#### Scenario: The prewarm evidence is incomplete or mismatched

GIVEN a missing boundary, a negative or regressing program-cache count, or a changed target-validation signature
WHEN the command evaluates the diagnostic result
THEN the command fails closed without selecting MTP, traced target decode, or production behavior

### Requirement: Prewarmed persistent recovery control preserves matched repeated evidence

r[tenstorrent.qwen38_p150x2.mtp_recovery_prewarmed_control]

The diagnostic P150x2 command MUST prewarm every configured prompt through nonpersistent target validation before persistent recovery warmups and measured roots. It MUST reuse the accepted persistent recovery profile, prompt set, route schedule, trace-capture count, exact state checks, stability gate, and recovery-ratio gate. It MUST atomically preserve ordered progress before and after each prewarm, warmup, measured root, evaluation, cleanup, and receipt boundary. Program-cache observations MUST be nonnegative and monotonic before measurement. The program-cache count MUST NOT change after measurement starts. Every warmup and measured target-validation signature MUST match its prompt prewarm signature. Missing progress, order drift, route drift, signature drift, program-cache drift, incomplete cleanup, or existing evaluator failure MUST fail closed. The command MUST remain admission-ineligible and MUST NOT change eager production behavior.

#### Scenario: Prewarmed repeated control completes

- **GIVEN** the pinned package, runtime, checkpoint, five prompts, balanced recovery schedule, and physical P150x2 devices
- **WHEN** the command prewarms each prompt, completes persistent warmups, and completes every measured root without program-cache growth
- **THEN** it SHALL write a diagnostic receipt with exact signatures, stable cache observations, complete recovery evidence, and the existing fail-closed decision

#### Scenario: Repeated control evidence is incomplete or changes

- **GIVEN** a missing boundary, order drift, route drift, signature mismatch, measured program-cache change, state mismatch, unstable root, or incomplete cleanup
- **WHEN** the command evaluates or preserves the control
- **THEN** it MUST reject the control without selecting MTP, traced target decode, or production behavior

### Requirement: Sampled MTP acceptance is distribution-exact and measured

r[tenstorrent.qwen38_p150x2.mtp_exact_rejection_sampling]

The repository MAY provide a pure, device-free MTP acceptance core for the pinned P150x2 runtime. For an argmax (point-mass) draft, the core MUST accept the draft with probability equal to its target probability `p(x̂)` when the exact rule is selected, so the emitted stream equals the target sampling distribution. It MUST report the realized per-position total-variation divergence `|a − p(x̂)|` for every rule. A lenient rule MUST reduce to the exact rule at lenience `1.0`. The greedy exact-match rule MUST remain available and MUST remain the production default. The sampled rules MUST remain diagnostic-only and MUST NOT change eager production behavior.

#### Scenario: Exact sampled acceptance preserves the target distribution

- GIVEN a point-mass draft and one per-row target candidate distribution
- WHEN the pure acceptance core resolves one round
- THEN each position reports its acceptance probability and realized divergence
- AND any rejected draft draws from the renormalised target residual

#### Scenario: Sampled acceptance fails closed on invalid input

- GIVEN a zero-length candidate list, a non-finite acceptance probability, a bounded lenience, or a rule identifier outside the closed set
- WHEN the pure acceptance core validates the round
- THEN the core MUST reject the round with a bounded reason
- AND it MUST NOT emit a token from an invalid distribution

### Requirement: MTP trace capture satisfies explicit hygiene gates

r[tenstorrent.qwen38_p150x2.mtp_trace_capture_hygiene]

The repository MUST validate MTP trace evidence with explicit pure capture-hygiene gates before the evidence is accepted. A control MUST hold at most one resident MTP trace at a time and MUST release a resident trace before capturing another. Capture MUST prime from real warm execution rounds, including one one-row seed round before a multi-row capture. Trace-bound buffers MUST be mutated in place at stable device addresses. The validation core MUST reject evidence from a capture that kept a second trace resident, captured from zeroed priming, or changed a trace-bound buffer address. The gates MUST remain pure and device-free and MUST NOT change eager production behavior.

#### Scenario: Capture hygiene passes

- GIVEN a control that released all resident traces, primed a one-row seed round, and captured with stable buffer addresses
- WHEN the pure hygiene gate validates the capture evidence
- THEN the gate MUST accept the trace evidence
- AND the evidence remains subject to the existing token, state, and parity gates

#### Scenario: Capture hygiene fails closed

- GIVEN a second trace resident during capture, zeroed priming, or a reallocated trace-bound buffer
- WHEN the pure hygiene gate validates the capture evidence
- THEN the gate MUST reject the evidence with a bounded reason
- AND the control MUST NOT accept a trace claim without an admitted provenance

### Requirement: GDN decode projection layouts tune only behind matched evidence

r[tenstorrent.qwen38_p150x2.gdn_decode_projection_layout]

The repository MAY provide an operator-enabled layout-tuning candidate for the pinned P150x2 GDN decode projections. The candidate MUST stay default-off, MUST NOT change prefill, and MUST leave the automatic layout path as production behavior. Its shape eligibility MUST live in a pure device-free core. Any speedup or selection claim MUST use matched eager-versus-candidate evidence under the existing decode-performance-admission requirement; a per-operation result alone MUST NOT imply a model speedup.

#### Scenario: Gated candidate is selected for diagnostic work

- GIVEN a matched physical eager-versus-candidate pair with stable medians
- WHEN the attribution comparator evaluates the gated candidate against the AUTO baseline
- THEN it MAY name the layout as a selected optimization candidate
- AND production keeps the automatic layouts until full admission evidence exists

#### Scenario: Candidate evidence is missing or unstable

- GIVEN a missing baseline root, identity drift, unstable medians, or an allocation failure under the gate
- WHEN the evaluator processes the candidate
- THEN it MUST reject the candidate and keep the automatic layout path selected

### Requirement: P150x2 decode serves concurrently with matched evidence

r[tenstorrent.qwen38_p150x2.batched_concurrency]

The repository MAY batch multiple active decode streams on the P150x2 service so
aggregate served tokens-per-second exceeds the single-stream floor. Batching MUST
run one full-model decode step per active batch of width `B <= 16`. Per-user
recurrent and KV state MUST stay isolated across steps and across users. Batch
selection MUST live in a pure device-free core. Any claim that batching raises
served throughput MUST use a matched concurrent-versus-single-stream physical pair
under the existing decode admission discipline; a per-op or per-layer result alone
MUST NOT imply a served-throughput gain.

#### Scenario: Concurrent streams are served in one step

- GIVEN `K` active users with stable positions and per-user state slices
- WHEN the scheduler assembles an active batch of width `B <= 16` and runs one decode step
- THEN each user advances exactly one token with its own state preserved
- AND the aggregate served tokens-per-second is measured against the single-stream floor

#### Scenario: State isolation or scheduling fails closed

- GIVEN an active batch that mixes two users, or a scheduler that selects outside `1..16`
- WHEN the evaluator processes the step
- THEN the core MUST reject the selection with a bounded reason
- AND the service MUST keep the single-stream path selected until matched evidence exists

### Requirement: P150x2 readiness follows bounded startup warm-up

r[tenstorrent.qwen38_p150x2.startup_readiness]

The Qwen3.8-27B P150x2 service MUST NOT report HTTP or systemd readiness before bounded startup warm-up succeeds. Warm-up MUST execute production prefill, exact-width trace capture, one trace replay, and clean reset for widths B=1 and B=4. Warm-up policy and lifecycle transitions MUST remain device-free. TTNN and systemd effects MUST remain in adapters or the service shell. Any warm-up, reset, or readiness-notification failure MUST abort startup. Health output MUST report the observed warm-up widths and elapsed time. Startup evidence MUST NOT imply physical admission or complete prompt-shape coverage.

#### Scenario: Warm-up completes before readiness

- GIVEN the physical P150x2 service starts with its pinned model and profile
- WHEN B=1 and B=4 warm-up complete with clean reset
- THEN the service MAY notify systemd and bind the ready HTTP endpoint
- AND `/healthz` and `/readyz` report the completed widths and elapsed time

#### Scenario: Warm-up fails closed

- GIVEN prefill, trace capture, trace replay, reset, or readiness notification fails
- WHEN startup evaluates the failure
- THEN the process MUST exit before it reports readiness
- AND the service MUST NOT accept an inference request

### Requirement: Batched throughput admission uses matched physical roots

r[tenstorrent.qwen38_p150x2.batch_performance_admission]

The repository MUST define batch admission in a closed typed profile. The profile MUST compare the same traced production path at widths B=1 and B=4. Physical execution MUST use devices 0 and 1 in exact `single,batch,batch,single` root order. Each root MUST use the same prompt corpus, generation count, execution artifact, runtime, checkpoint, tokenizer, mesh, and immutable execution policy. The execution identity MUST bind all code, patches, runtime dependencies, and policy that can change measured behavior. It MUST exclude only receipt-binding metadata that cannot change execution. The final deployed closure MUST reference the measured execution identity before `physically_admitted` can become true. Admission MUST require exact per-request token parity, a complete active state prefix, distinct state slices, clean reset, aggregate decode speedup at least `3.0`, per-user decode-rate retention at least `0.75`, TTFT ratio at most `4.5`, and tail-latency ratio at most `1.5`. The existing single-stream admission profile and its `1.05` TTFT gate MUST remain unchanged.

#### Scenario: B=4 passes every batch gate

- GIVEN complete current-identity B=1 and B=4 physical roots with matching requests
- WHEN the deterministic batch receipt evaluator checks correctness, isolation, timing, throughput, cleanup, and execution identity
- THEN it MUST admit only when every profile-owned bound passes
- AND the admitted profile MUST bind the receipt BLAKE3 identity.

#### Scenario: Receipt metadata is promoted

- GIVEN a passing receipt for the immutable execution identity
- WHEN only the receipt hash and admission decision change
- THEN the execution identity MUST remain unchanged
- AND the final deployed closure MUST reference that same identity and receipt.

#### Scenario: An execution input changes

- GIVEN a measured execution identity
- WHEN code, patches, runtime dependencies, or immutable execution policy change
- THEN the execution identity MUST change
- AND the old receipt MUST remain ineligible for admission.

### Requirement: Batch TTFT excludes trace preparation

r[tenstorrent.qwen38_p150x2.batch_ttft_evidence]

Batch TTFT MUST start immediately before production prefill dispatch and stop after every active user's first-token winner is available. Exact-width trace preparation MUST occur after that boundary and MUST remain in a separate evidence field. Queue collection, later decode steps, and cleanup MUST NOT enter the internal TTFT phase. Normal production and soak responses MUST remain admission-ineligible. Only an explicit operator batch-admission mode MAY emit admission-eligible evidence.

#### Scenario: Batch phase evidence is complete

- GIVEN an explicit batch-admission run with valid B=1 or B=4 requests
- WHEN prefill, first-token selection, trace preparation, decode, and cleanup complete
- THEN each response MUST report finite non-negative separated phase durations
- AND its TTFT MUST exclude trace preparation while retaining first-token work.

#### Scenario: Ordinary traffic reaches the receipt evaluator

- GIVEN a normal production, readiness, soak, profiled, or simulated response
- WHEN the batch receipt evaluator reads it
- THEN it MUST reject the response as admission-ineligible
- AND it MUST not infer admission from a throughput value.

### Requirement: Same-width groups reuse at most one decode trace

r[tenstorrent.qwen38_p150x2.same_width_trace_reuse]

The service MAY reuse one resident decode trace across consecutive request groups of the same exact width. Reuse MUST require a complete prior reset and stable token, position, RoPE, page-table, KV, GDN, output, and trace bindings. The service MUST release the resident trace before prefill when the width changes, policy disables reuse, reset is incomplete, binding drifts, or more than one trace exists. The service MUST NOT keep two decode traces resident. Production reuse MUST remain disabled until matched physical evidence passes every profile-owned correctness, identity, stability, and latency bound. A passing reuse result MUST NOT admit slot recycling, mixed generation bounds, dynamic compaction, or live request insertion.

#### Scenario: A same-width group reuses one stable trace

- GIVEN one resident trace, the same requested width, a complete reset, and stable bindings
- WHEN the next group completes prefill and prepares decode
- THEN the service MUST reuse the existing trace without another capture
- AND the response evidence MUST report one resident trace and exact output parity.

#### Scenario: Reuse eligibility fails

- GIVEN a width change, disabled policy, incomplete reset, binding drift, or multiple resident traces
- WHEN the service plans the next prefill
- THEN it MUST release every resident trace before prefill
- AND it MUST capture at most one new trace after prefill.

#### Scenario: Physical reuse evidence fails

- GIVEN missing identity, token mismatch, reset failure, trace-count drift, unstable timing, or insufficient wall-time improvement
- WHEN the deterministic evaluator reads the matched roots
- THEN it MUST reject production reuse with bounded reasons
- AND the service MUST preserve the admitted recapture path.
