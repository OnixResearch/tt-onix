# Complete the RWKV-7 fused WKV readback

## Why

The diagnostic-only BF16 readout runtime enqueues its three fused kernels, but synchronized result readback does not finish. The last physical run that completed used the earlier FP32 readout path. We must locate the first incomplete device stage, correct the fused path, and obtain exact asymmetric physical parity before any new window-16 evaluation.

## What Changes

- Add bounded device-stage evidence for recurrence, BF16 readout, correction, output emission, and writer completion.
- Correct the proven precision, readout, and correction-geometry defects without changing production admission.
- Require exact FP32 state and exact BF16 row parity against package-matched generic TT-NN.
- Add positive and negative source contracts for precision, ownership, signaling, row geometry, and diagnostic bounds.
- Preserve the completed diagnostic result for a separate immutable window-16 admission change.

## Impact

- **Files**: fused WKV host program, three fused kernels, evidence contracts, Cairn artifacts, and generated quality evidence.
- **Testing**: focused Rust contracts, device-free runtime self-tests, exact diagnostic package builds, bounded physical stage probes, exact physical parity, Nix closure checks, formatting, Clippy, repository quality, and Cairn gates.

## Non-goals

- This change does not weaken exact parity.
- This change does not reuse the rejected candidate identity.
- This change does not claim physical success from enqueue, timeout, simulator, or partial output.
- Production windows remain `[2, 4, 8]` unless a separate immutable candidate passes every admission gate.
