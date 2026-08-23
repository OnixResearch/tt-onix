## Why

The exact prompt-2 control completed after target-validation prewarm. The full persistent recovery control still lacks complete repeated roots and a receipt.

## What Changes

- Add one diagnostic mode that prewarms every fixed prompt before persistent recovery warmups.
- Preserve atomic progress across prewarm, warmup, measured roots, evaluation, cleanup, and receipt creation.
- Reuse the existing five-prompt balanced route schedule and fail-closed recovery evaluator.
- Require exact per-prompt signatures and no measured program-cache growth.
- Keep production eager and admission-ineligible.

## Impact

- **Files**: Qwen recovery control core, command shell, package wrapper, checks, specification, and adoption documentation.
- **Testing**: Positive and negative device-free tests, Qwen package checks, service checks, Cairn validation, repository-quality checks, and one separately authorized bounded physical control.
