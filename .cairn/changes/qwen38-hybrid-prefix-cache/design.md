## Context

A Qwen3.8 prefix spans both attention families. The September report derives 72 MiB of raw recurrent state per chip, before convolution and KV. At 2048 tokens, raw BF16 KV adds 64 MiB per chip.

## Decisions

Start with one approved shared system prefix and a bounded in-process cache. Do not persist device addresses or snapshots across restarts.

The core owns a content-bound key, completeness checks, memory accounting, tenant quotas, and eviction order. The shell supplies authenticated tenant scope and measured allocation facts. The TTNN adapter owns snapshot and restore effects.

The key includes exact tokens and all identity inputs from the evidence contract. Similar text is not a match. The entry lists recurrent state, convolution carry, full-attention KV, positions, and any final hidden row required by the continuation route.

Entries are immutable. A continuation receives isolated state through copies or a proven copy-on-write contract. Partial KV blocks require separate ownership before writes. No cached prefix contains live MTP speculation state in the first version.

Stable destination buffers belong to the current trace plan. A restore cannot replace trace-bound allocations. A digest proves content identity, not authorization or correct state.

## Validation

Positive tests cover an identical prefix, an empty suffix, and repeated continuation under the same route. Negative tests cover a tokenizer change, head replacement, tenant mismatch, missing recurrent state, wrong tensor shape, corrupted digest, partial restore, and exhausted memory.

Matched hardware compares cold prefill with lookup plus full restore. It includes cache misses and eviction. A useful hit must improve complete request latency without a failed memory or quality gate.

## Alternatives and rollback

A KV-only cache is rejected. Unbounded per-conversation snapshots are rejected. Cross-restart persistence and automatic cross-tenant sharing are deferred.

On a miss, use cold prefill. On failed restoration, reset or quarantine before cold fallback. The feature stays opt-in until the exact profile passes.
