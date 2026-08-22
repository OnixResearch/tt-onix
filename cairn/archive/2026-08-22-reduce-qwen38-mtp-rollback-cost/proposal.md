## Why

The rejected MTP pilot rebuilds the complete 128-token MTP cache after a target rejection. The latest trace control showed correct rollback and positive aggregate opportunity, but prompt 2 still paid MTP re-prefill after accepting zero drafts. Multi-round MTP needs an exact and cheaper way to restore rejected cache rows and advance the committed target token.

## What Changes

- Add a diagnostic-only device-resident MTP K/V row snapshot and restore path.
- Compare cache re-prefill with snapshot restore plus committed-token advance in balanced matched roots.
- Fail closed on state mismatch, route drift, unstable recovery timing, insufficient recovery gain, or invalid evidence.
- Keep production eager and keep production wrappers from loading `mtp.*` weights.

## Impact

- **Files**: Qwen MTP rollback policy, pure evaluator, TTNN adapter, diagnostic command, package checks, evidence, and adoption documentation.
- **Testing**: positive and negative policy tests, Nickel export, package checks, matched P150x2 evidence, Cairn gates, and repository artifact gates.
