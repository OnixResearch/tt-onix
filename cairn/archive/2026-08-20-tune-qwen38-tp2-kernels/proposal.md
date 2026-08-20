## Why

TP2 tuning needs an attributed operation profile before any bounded sweep can select candidates.
No complete TP2 operation profile, legal sweep matrix, or selected configuration set exists.

Without those inputs, a deterministic winner and an end-to-end admission result cannot be produced.

## Outcome

Return TP2 kernel tuning as unselected and unadmitted.
Keep the current Qwen3.8 TP2 configurations and safe fallbacks unchanged.
Do not describe them as TP2-tuned.

This result does not reject TP2 tuning on technical merit.
It closes this sweep because no profiled candidate set exists.

## What Changes

- Record the current TP2 fallback behavior and missing profiler evidence.
- Keep all current matrix and collective configurations unchanged.
- Add no generated TP2 winner data or model speedup claim.
- Require a new validated Cairn change for future profiling and tuning.
- Require operation attribution before any physical sweep in that change.

## Impact

- **Model package**: no configuration patch changes.
- **Runtime**: current TP2 compatibility behavior remains selected.
- **Configuration**: no TP2 tuning profile or generated winner is added.
- **Evidence**: the archive records the missing profile and candidate set.
- **Documentation**: existing configurations are not called TP2-tuned.

## Out of Scope

- Guessing grids from TP4 or TP8 results.
- Selecting candidates from static shape inspection.
- Claiming model gains from one operation.
- Changing precision, runtime, or batching.
