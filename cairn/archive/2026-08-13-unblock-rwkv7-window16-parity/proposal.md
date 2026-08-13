# Unblock the RWKV-7 window-16 parity diagnostic

## Why

The prior physical attempts did not prove the BF16 readout correction. The retained watcher data shows that the probe compiled a large generic TT-NN reference graph before comparison. The retained launch symlink also resolves to the rejected candidate runtime, not the diagnostic runtime.

We need a bounded probe that binds its package identity, reaches the fused kernel without generic device-side pre-work, and writes an exact parity receipt.

## What changes

- Replace the asymmetric probe's generic TT-NN device reference with a pure deterministic host reference over the same BF16 and FP32 fixture values.
- Bind each probe receipt to the compiled runtime name and BF16 readout setting.
- Add explicit bounded stage markers that distinguish fixture setup, fused compilation, execution, readback, and comparison.
- Add positive and negative source and reference tests for package binding and generic-device-prework removal.
- Run one bounded physical diagnostic on device `0` with the exact diagnostic package.
- Start a new immutable window-16 evaluation only if exact FP32 state and BF16 corrected-row parity pass.

## Non-goals

- This change does not weaken exact parity.
- This change does not admit the rejected candidate.
- This change does not change production windows `[2, 4, 8]` before a complete new evaluation.
- This change does not use simulator evidence for physical correctness or performance claims.
