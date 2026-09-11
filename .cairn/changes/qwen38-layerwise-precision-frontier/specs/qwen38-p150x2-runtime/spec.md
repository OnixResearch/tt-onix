## ADDED Requirements

### Requirement: Explicit precision recipe and cache identity

r[tenstorrent.qwen38_p150x2.precision_recipe_identity]

Typed policy MUST name each changed weight family, layer set, compute fidelity, accumulation type, and KV format. Initial candidates MUST retain FP32 recurrent state and the BF16 MTP head. Each recipe MUST use a separate content-bound cache namespace. Checkpoint FP8 MUST NOT be treated as TT BFP8.

#### Scenario: Valid down-projection recipe
- GIVEN a recipe that changes only selected MLP down weights
- WHEN the loader creates a candidate cache
- THEN it records the exact recipe and leaves control caches unchanged.

#### Scenario: Stale cache or sensitive head change
- GIVEN a mismatched cache identity or an unauthorized MTP-head precision change
- WHEN the loader validates the recipe
- THEN it rejects the candidate before device use.

### Requirement: Decode and state quality coverage

r[tenstorrent.qwen38_p150x2.precision_quality]

The quality evaluator MUST cover prefill scores, teacher-forced decode logits, long carry within the admitted context, and chunk-boundary invariants. It MUST report task-quality results and MTP acceptance where the shared head changes. Greedy exactness and sampled distributional gates MUST remain distinct.

#### Scenario: Complete quality screen
- GIVEN a candidate with finite outputs and matched reference inputs
- WHEN all declared quality gates pass
- THEN the receipt records the supported route and measured quality scope.

#### Scenario: Prefill pass but decode drift
- GIVEN a candidate with a passing prefill score and a failing decode or state gate
- WHEN quality admission runs
- THEN it rejects the candidate despite the prefill score.

### Requirement: Measured speed-memory-quality frontier

r[tenstorrent.qwen38_p150x2.precision_frontier]

The evaluator MUST report actual cache bytes, device memory, complete decode time, request latency, and quality for every recipe. BF8 KV MUST have a separate context-dependent screen. A recipe MUST NOT become a default without matched evidence and explicit owner approval of its quality trade-off.

#### Scenario: Useful selective precision
- GIVEN a recipe that meets every quality and performance bound
- WHEN the owner reviews its measured trade-off
- THEN only the approved recipe and scope can enter production admission.

#### Scenario: Smaller weights without speed
- GIVEN reduced cache bytes but failed latency or quality gates
- WHEN the evaluator ranks recipes
- THEN it retains the result as rejected evidence and keeps the control default.
