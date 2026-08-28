## ADDED Requirements

### Requirement: Sampled-mode kernel admission uses bounded distributional evidence

r[tenstorrent.qwen38_p150x2.sampled_mode_kernel_admission]

The repository MAY admit a default-off decode execution-identity candidate for sampled serving only, under a closed typed distributional admission policy. The policy MUST stay pure and device-free and MUST own every admission bound. Probe evidence MUST come from teacher-forced decode on matched physical roots in exact `control,candidate,candidate,control` order over a fixed corpus of at least five prompts and forty-eight positions each. Admission MUST require mean shared-token logit PCC, mean and maximum truncated top-K total variation, bounded mean top-K residual mass, margin-stratified argmax agreement on stable positions, state isolation, clean reset, and the existing GDN and full decode-step timing bounds. The per-position minimum logit PCC MUST be reported as a diagnostic and MUST NOT gate the verdict. Positions whose control top1-top2 logit margin is below the fixed near-tie epsilon MUST be reported without gating the verdict. A candidate admitted under this requirement MUST serve sampled requests only. Greedy and deterministic requests MUST keep the exact-parity path and the previously admitted execution identity. No accuracy-benchmark claim MAY be derived from this evidence.

#### Scenario: Sampled-mode distributional evidence admits a candidate

r[tenstorrent.qwen38_p150x2.sampled_mode_kernel_admission.valid]

- **GIVEN** a complete teacher-forced probe corpus on matched physical roots where every PCC, total-variation, residual-mass, stable-agreement, state, reset, and timing bound passes
- **WHEN** the pure distributional admission core evaluates the evidence
- **THEN** the receipt MAY make the candidate eligible for sampled serving only
- **AND** deployment MUST wait for fresh B=4 admission of the changed execution identity.

#### Scenario: Distributional evidence is invalid, short, or below any bound

r[tenstorrent.qwen38_p150x2.sampled_mode_kernel_admission.invalid]

- **GIVEN** a missing metric, a corpus below its floor, an unbalanced root order, a non-finite value, disjoint top-K sets, a missed bound, or an unstable stable-position agreement
- **WHEN** the pure distributional admission core evaluates the evidence
- **THEN** it MUST reject the candidate with a bounded reason
- **AND** production MUST keep the previously admitted execution identity for every serving mode.

## MODIFIED Requirements

### Requirement: Decode speedup claims require matched physical evidence

r[tenstorrent.qwen38_p150x2.decode_performance_admission] A traced candidate MUST match the eager path's greedy generated tokens for the profile corpus. A speedup claim MUST use order-balanced physical A/B evidence. The evidence MUST bind both paths to the same closure, checkpoint, tokenizer, workload, mesh, and generation settings. Nickel MUST own every admission bound. A candidate admitted for sampled serving under the sampled-mode kernel admission requirement MUST NOT serve greedy or deterministic requests, and its speedup claim MUST cite the distributional admission receipt instead of greedy token parity.

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

#### Scenario: A sampled-mode candidate receives a greedy request

r[tenstorrent.qwen38_p150x2.decode_performance_admission.sampled_only]
- GIVEN a candidate admitted only under the sampled-mode kernel admission requirement
- WHEN a greedy or deterministic request arrives
- THEN the service MUST route that request to the exact-parity execution identity
- AND the distributional receipt MUST NOT authorize deterministic output from the candidate.
