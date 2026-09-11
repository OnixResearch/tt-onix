## ADDED Requirements

### Requirement: Content-bound experiment identity

r[tenstorrent.qwen38_p150x2.experiment_identity]

The experiment evaluator MUST bind the backbone, conversion manifest, MTP head, tokenizer, template, cache recipe, execution closure, route, and physical mesh. The shell MUST compute content identities with BLAKE3 and retain source revisions. A path name MUST NOT substitute for content identity.

#### Scenario: Matched hybrid checkpoint
- GIVEN a complete manifest for an FP8 backbone and BF16 MTP head
- WHEN both comparison roots report that manifest
- THEN the evaluator accepts the identity comparison without claiming performance success.

#### Scenario: Renamed or replaced head
- GIVEN the same checkpoint directory with different MTP bytes
- WHEN a root presents the old manifest
- THEN the evaluator rejects the root before admission.

### Requirement: Complete and comparable timing

r[tenstorrent.qwen38_p150x2.experiment_timing]

The evaluator MUST distinguish client latency, internal first-token readiness, prefill, capture, decode, selection, commit, rollback, and cleanup. MTP economics MUST include every round phase. Diagnostic instrumentation MUST make a root admission-ineligible. Missing roots or failed stability gates MUST NOT disappear from the comparison.

#### Scenario: Complete balanced control
- GIVEN matched roots with complete timing and a predeclared workload
- WHEN the evaluator calculates performance
- THEN it reports aggregate emitted-token cost and every declared gate separately.

#### Scenario: Incomplete fast result
- GIVEN a result that omits rollback or contains only a profiled candidate
- WHEN the evaluator calculates admission
- THEN it rejects the result with a specific evidence error.

### Requirement: Bounded experiment authority

r[tenstorrent.qwen38_p150x2.experiment_authority]

The shell MUST require explicit outage authority before hardware ownership changes. It MUST bind the expected listener, device IDs, execution identity, deadline, and recovery procedure. A timed-out device worker MUST block reuse and concurrent cleanup. Simulator receipts MUST state that they are simulated.

#### Scenario: Authorized diagnostic
- GIVEN an approved window and an unused diagnostic listener
- WHEN the shell starts the bounded candidate
- THEN one worker owns the expected mesh and the receipt names its exact identity.

#### Scenario: Foreign listener or timeout
- GIVEN a foreign service on the diagnostic listener or a stalled device worker
- WHEN the shell observes that condition
- THEN it rejects admission and prevents another device request.
