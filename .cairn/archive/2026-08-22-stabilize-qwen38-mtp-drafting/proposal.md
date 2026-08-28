## Why

The paired MTP selector control removed full-vocabulary readback, but complete speculative generation slowed to `0.8197972276186697x`. MTP draft compute remained unstable, including a `0.4443297780235298`-second prefill outlier. Package-matched source shows that each draft decode has fixed shapes, explicit position inputs, and internally addressed TP attention caches. This makes persistent trace replay the smallest remaining way to stabilize and reduce draft cost before confidence gating.

## What Changes

- Add a typed diagnostic profile for matched eager and traced MTP draft-step replay.
- Add a deterministic evaluator that rejects unstable, incomplete, mismatched, or uneconomic evidence.
- Add a narrow TTNN trace adapter with stable token, position, RoPE, feedback, cache, and output bindings.
- Compare exact draft tokens, target tokens, accepted lengths, rollback results, and bounded MTP KV-prefix state across matched routes.
- Keep production eager and keep MTP diagnostic-only.

## Impact

- **Files**: `pkgs/qwen36`, Qwen adoption documentation, Cairn evidence, and the generated artifact manifest
- **Testing**: positive and negative policy tests, Ruff, Python compilation, Nickel export, Qwen package checks, Cairn validation, repository artifact gates, and one authorized matched P150x2 control
