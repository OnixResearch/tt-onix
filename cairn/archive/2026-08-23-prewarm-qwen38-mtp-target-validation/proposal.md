## Why

The staged physical probe completed both MTP recovery routes before it reached its timeout. The re-prefill route stopped in native target verification. The snapshot route stopped in target reference-state construction. Both blocked paths compiled target programs after persistent MTP trace capture.

A same-process exact prewarm can separate cold target compilation from parked-trace coexistence. The control must preserve its own progress and remain admission-ineligible.

## What Changes

- Add a diagnostic stage command that prewarms the same prompt through the existing nonpersistent target-validation path.
- Preserve prewarm and measured boundaries with program-cache entry counts in the atomic progress ledger.
- Require exact prewarm and measured target-validation signatures before receipt creation.
- Keep the existing stage command and eager production service unchanged.

## Impact

- **Files**: Qwen MTP stage core, drafter shell, package wrappers, device-free checks, Cairn evidence, and adoption documentation.
- **Testing**: Positive and negative pure-core tests, Python compilation, Ruff, Qwen checks, NixOS service checks, Cairn validation, and repository-quality checks.
