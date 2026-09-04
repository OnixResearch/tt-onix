## ADDED Requirements

### Requirement: Exact compact greedy selection

r[tenstorrent.qwen38_p150x2.compact_greedy_selection]

The verifier adapter MUST reduce logits before host readback on the compact route. The pure merge MUST return the same global token as full-logit argmax. It MUST resolve equal logits by the declared global-ID rule. The adapter MUST preserve all verifier continuation rows and state semantics.

#### Scenario: Winner on another shard
- GIVEN finite logits whose maximum lies on the second physical shard
- WHEN the compact verifier selects a token
- THEN its global token ID equals the full-logit reference.

#### Scenario: Ties, padding, or invalid values
- GIVEN tied values across shards, padded vocabulary entries, or non-finite logits
- WHEN selection occurs
- THEN the tie follows the reference rule, padding cannot win, and non-finite input produces an explicit error.

### Requirement: Bounded sampled candidate coverage

r[tenstorrent.qwen38_p150x2.compact_sampled_selection]

The sampled candidate MUST preserve the existing top-K, temperature, nucleus, seed, and first-token contract. Local index ranges MUST fit the selected kernel. Global offsets MUST use a wide integer representation. Unsupported penalties or insufficient candidate coverage MUST select the named fallback before sampling.

#### Scenario: Exact candidate merge
- GIVEN complete local top-K sets under the supported transforms
- WHEN the core merges them with an injected uniform draw
- THEN the selected token equals the full-logit reference under the same draw.

#### Scenario: Unsupported index or transform
- GIVEN a shard beyond the kernel index range or an unproved penalty transform
- WHEN the adapter selects a reduction plan
- THEN it uses a proven sub-shard plan or rejects compact execution.

### Requirement: Separate selection admission

r[tenstorrent.qwen38_p150x2.compact_selection_admission]

The evaluator MUST compare greedy and sampled candidates separately. It MUST record device reduction, transfer, host merge, complete round time, and request latency. Full-logit diagnostics MUST remain available without entering the compact performance sample. Production promotion MUST require matched correctness and performance evidence.

#### Scenario: Compact path passes
- GIVEN matched compact and reference roots with unchanged model state
- WHEN all route-specific gates pass
- THEN the receipt admits only the measured route and identity.

#### Scenario: Small payload but slower service
- GIVEN a smaller readback whose complete request regresses
- WHEN the evaluator applies the predeclared gates
- THEN it rejects the candidate despite the byte reduction.
