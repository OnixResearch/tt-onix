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

### Requirement: Continuous batching remains unsupported without an admitted execution base
r[tenstorrent.qwen38_p150x2.continuous_batching_non_support] The repository MUST keep Qwen3.8-27B P150x2 request execution serialized while its required traced execution base is unadmitted. It MUST NOT add or claim a continuous-batching scheduler, batched state path, batch profile, or aggregate throughput result from single-request evidence. Future continuous-batching support MUST use a new validated Cairn change with an admitted execution base and complete matched physical evidence.

#### Scenario: Traced execution base is unadmitted
r[tenstorrent.qwen38_p150x2.continuous_batching_non_support.valid]
- GIVEN the required traced execution base has a rejected physical receipt
- WHEN continuous-batching support is evaluated
- THEN the service MUST remain serialized
- AND no continuous-batching implementation or throughput claim MUST be added.

#### Scenario: A future traced path is admitted
r[tenstorrent.qwen38_p150x2.continuous_batching_non_support.invalid]
- GIVEN an execution base becomes admitted after this decision
- WHEN an operator requests continuous batching
- THEN this archived decision MUST NOT imply batch support
- AND a new validated change with batch-specific physical evidence MUST be required.

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
