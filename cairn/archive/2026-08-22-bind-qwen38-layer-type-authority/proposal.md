## Why

The adopted Qwen source still declares an unused eight-entry `FULL_ATTENTION_LAYERS` set for the older 32-layer mapper. Qwen3.8 runtime dispatch instead reads the pinned checkpoint's 64-entry `layer_types` list. The source does not make that authority boundary explicit, and one archived host-attribution result also reverses the named eager and candidate receipt fields.

## What Changes

- Add a pure layer-type resolver that accepts only a complete recognized checkpoint list.
- Keep the 32-layer legacy fallback only when the declared layer count matches it.
- Remove the unused mapper constant so it cannot be mistaken for runtime authority.
- Add positive and negative device-free tests for 64-layer Qwen3.8 dispatch and malformed inputs.
- Record the source dataflow and correct the reversed historical evidence labels.

## Impact

- **Files**: Qwen source patch, package checks, adoption documentation, and one archived evidence narrative
- **Testing**: Qwen package gate, authority negative gate, Python checks, Cairn validation, and focused repository-quality gates
