## Why

The persistent MTP recovery control produced no receipt before its two-hour bound. The result does not identify the blocked host or device stage.

## What Changes

- Add a one-prompt, one-route diagnostic mode with a durable stage ledger.
- Record progress outside measured timing windows before and after each replay boundary.
- Run short bounded P150x2 cases and preserve the last completed stage on timeout.
- Keep production eager and make the staged result admission-ineligible.

## Impact

- **Files**: Qwen MTP diagnostic core, adapters, command, package wrapper, checks, Cairn evidence, and adoption documentation
- **Testing**: Positive and negative stage-transition tests, existing Qwen checks, package checks, and one authorized bounded physical control
