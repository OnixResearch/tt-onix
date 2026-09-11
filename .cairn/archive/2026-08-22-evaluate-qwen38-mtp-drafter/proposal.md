## Why

The exact-P150x2 traced target verifier is valid and provides a `0.11282991201733239`-second width-four budget for drafting and acceptance. The checkpoint contains one MTP decoder layer, but the packaged loader discards all `mtp.*` weights. Target feasibility alone cannot establish speculative speedup.

## What Changes

- Bind the checkpoint MTP equation, weights, source authorities, and shared embedding and LM-head behavior.
- Add an explicit diagnostic drafter path that loads only the pinned MTP weights and leaves production eager.
- Measure greedy draft acceptance, draft cost, target-verifier cost, emitted-token economics, and exact target output on P150x2.
- Implement or reject bounded rollback for target GDN state, target paged KV, and MTP attention KV.
- Fail closed when mapping, parity, acceptance, rollback, stability, or end-to-end economics is incomplete.

## Impact

- **Files**: Qwen36 diagnostic policy, MTP loader and adapters, target prefix-state support, tests, evidence, adoption documentation, and the Qwen38 P150x2 runtime specification.
- **Testing**: Positive and negative pure-policy tests, deterministic weight-shape checks, package checks, focused Nix quality gates, and bounded physical P150x2 evidence.
