## ADDED Requirements

### Requirement: Complete prefix identity and authority

r[tenstorrent.qwen38_p150x2.hybrid_prefix_identity]

The prefix key MUST bind exact token IDs, prefix length, model content, execution closure, tokenizer, template, precision, and tenant scope. BLAKE3 MUST identify new content-addressed entries. A cache hit MUST NOT grant cross-tenant access or accept a changed checkpoint.

#### Scenario: Authorized identical prefix
- GIVEN an identical prefix under the same admitted identity and tenant scope
- WHEN lookup finds a complete entry
- THEN the core returns a restore plan within the caller's authority.

#### Scenario: Identity or tenant mismatch
- GIVEN matching text with different token IDs, precision, model bytes, or tenant scope
- WHEN lookup runs
- THEN the core returns a miss or denial without exposing another entry.

### Requirement: Complete hybrid restore

r[tenstorrent.qwen38_p150x2.hybrid_prefix_state]

An entry MUST include all required full-attention KV, recurrent state, convolution state, and position facts. Required hidden state MUST have an explicit role. Restore MUST isolate mutable continuation state and preserve trace-address constraints. KV-only entries MUST fail validation.

#### Scenario: Restored continuation
- GIVEN a complete immutable entry and stable destination buffers
- WHEN the adapter restores the prefix and runs continuation
- THEN tokens and hybrid state match the cold-prefix reference.

#### Scenario: Partial or corrupted entry
- GIVEN a missing recurrent tensor, wrong shape, bad digest, or partial copy
- WHEN restore validation runs
- THEN the adapter rejects the entry and prevents continuation from partial state.

### Requirement: Bounded memory and measured cache benefit

r[tenstorrent.qwen38_p150x2.hybrid_prefix_admission]

Typed policy MUST bound entry count, memory bytes, tenant quotas, and eviction. The evaluator MUST include lookup, restore, copy-on-write, miss, and eviction costs. Admission MUST require matched request-wall benefit without an unapproved quality or memory trade-off.

#### Scenario: Useful bounded cache
- GIVEN a repeated-prefix workload within the memory budget
- WHEN matched warm and cold controls pass all gates
- THEN the receipt admits only that bounded cache profile.

#### Scenario: Eviction or negative benefit
- GIVEN an exhausted budget or restore cost above avoided prefill
- WHEN policy evaluates the request
- THEN it evicts safely or uses cold prefill without claiming a speedup.
