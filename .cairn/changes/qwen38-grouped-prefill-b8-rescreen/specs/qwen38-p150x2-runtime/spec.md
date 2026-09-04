## ADDED Requirements

### Requirement: Derived grouped-prefill eligibility

r[tenstorrent.qwen38_p150x2.grouped_prefill_eligibility]

The pure planner MUST derive the group bound from runtime capability, local value heads, bucket shape, worker grid, and memory budget. It MUST NOT copy a TP4 group cap into TP2. Unsupported groups MUST use the current per-user staged route.

#### Scenario: Supported short-prompt group
- GIVEN an exact TP2 capability record and a group within every bound
- WHEN the planner evaluates the group
- THEN it returns a grouped plan with explicit tensor shapes and state destinations.

#### Scenario: Head or memory overflow
- GIVEN a group whose head rows or scratch memory exceed a bound
- WHEN the planner evaluates the group
- THEN it selects per-user prefill before any device allocation.

### Requirement: Isolated valid-length state

r[tenstorrent.qwen38_p150x2.grouped_prefill_state]

Grouped prefill MUST preserve every user's valid-length mask, full-attention KV, recurrent state, and convolution state. Padding MUST NOT advance the committed prefix. Partial failure MUST quarantine or reset the affected group before reuse.

#### Scenario: Different prompt lengths
- GIVEN different prompts within one eligible bucket
- WHEN grouped prefill and continuation complete
- THEN each row matches its independent per-user reference under the declared numerical contract.

#### Scenario: Padded or failed user
- GIVEN a short padded prompt or a failed state installation
- WHEN the adapter commits the group
- THEN it excludes padding from state and rejects incomplete installation.

### Requirement: B4 preservation and separate B8 decision

r[tenstorrent.qwen38_p150x2.grouped_prefill_admission]

The evaluator MUST first preserve the admitted B4 contract on the new execution identity. A B8 screen MUST retain the existing B8 correctness, aggregate speed, per-user retention, TTFT, and tail gates. Lower prefill cost MUST NOT substitute for decode retention.

#### Scenario: Complete B8 improvement
- GIVEN grouped prefill with complete matched B8 roots
- WHEN every unchanged B8 gate passes
- THEN the receipt admits B8 only for that identity and workload.

#### Scenario: TTFT-only improvement
- GIVEN improved B8 TTFT but failed decode retention
- WHEN the evaluator applies the B8 policy
- THEN B8 remains unadmitted and B4 remains separately governed.
