## ADDED Requirements

### Requirement: Isolated immutable runtime candidate

r[tenstorrent.qwen38_runtime.candidate_077_identity]

The candidate MUST start from TT-Metal revision `9f9cd4fd590f4b606bd0981a4fe0b6403eb38ec9`. Its manifest MUST bind compatible model source, local patches, SFPI, UMD, KMD, simulator, and Python dependencies. The repository MUST retain the 0.74 control. Nix MUST generate any lockfile changes.

#### Scenario: Complete candidate closure
- GIVEN an immutable candidate and an explicit dependency manifest
- WHEN Nix builds the candidate package
- THEN it produces a separate closure without changing the production default.

#### Scenario: Mixed runtime components
- GIVEN model code or a device kernel from an unbound runtime revision
- WHEN the capability evaluator inspects the closure
- THEN it rejects the candidate as incompatible or incomplete.

### Requirement: State-safe fused prefill

r[tenstorrent.qwen38_runtime.candidate_077_prefill]

The fused candidate MUST preserve Q/K normalization, FP32 recurrent state, valid-length masking, convolution carry, and final-state ownership. It MUST preallocate constant tiles before trace capture. Each removed compatibility fallback MUST have an exact TP2 capability test.

#### Scenario: Full chunk and masked tail
- GIVEN a real prefix followed by a partial chunk
- WHEN fused prefill finishes and decode continues
- THEN logits and hybrid state satisfy the declared reference gates at each boundary.

#### Scenario: Missing capability or unsafe mask upload
- GIVEN a missing operation signature or host upload during trace capture
- WHEN candidate preflight runs
- THEN it rejects the candidate and retains the named control fallback.

### Requirement: Bounded uplift decision

r[tenstorrent.qwen38_runtime.candidate_077_admission]

The evaluator MUST require a representative simulator result or an exact simulator blocker before authorized hardware evaluation. Hardware evidence MUST bind the candidate closure and exact P150x2 mesh. Failed correctness, memory, latency, or stability gates MUST leave the runtime unadmitted.

#### Scenario: Matched physical comparison
- GIVEN a valid candidate and complete control roots
- WHEN all predeclared gates pass
- THEN the receipt admits only the tested capabilities and workload.

#### Scenario: Version-only success
- GIVEN a successful import or simulator run without matched hardware evidence
- WHEN the evaluator considers performance admission
- THEN it withholds every silicon speed claim.
