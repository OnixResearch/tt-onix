## Why

The exact P150x2 MTP pilot missed aggregate break-even by `3.1278%`, even though four of five prompts accepted all three drafts. Source review found that every MTP draft gathers the full vocabulary across devices and reads the full replicated logit row to the host. The exact target trace already proves that shard-local argmax with bounded winner readback preserves greedy token selection.

## What Changes

- Compare full-logit MTP draft selection with shard-local top-one selection in one loaded process.
- Reuse the target path's deterministic cross-shard tie policy instead of adding a second greedy policy.
- Measure draft selection separately from MTP compute, target verification, rollback, and cache rebuild.
- Require exact draft-token, target-token, state, and rollback parity across both selector routes.
- Select no production change unless the bounded selector is stable, materially faster, and moves emitted-token economics above break-even.

## Impact

- **Files**: Qwen diagnostic policy, pure selection evaluator, shared shard-winner adapter boundary, MTP drafter probe, tests, evidence, and adoption documentation.
- **Runtime**: Physical P150x2 only, batch one, three drafts, fixed five-prompt suite, diagnostic-only.
- **Production**: The eager service and production wrappers remain unchanged.
