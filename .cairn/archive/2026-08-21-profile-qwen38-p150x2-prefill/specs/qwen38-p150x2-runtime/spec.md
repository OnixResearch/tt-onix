## ADDED Requirements

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
