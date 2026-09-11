# Resolve the RWKV-7 window-16 parity failure

## Why

The bounded window-16 evaluation stopped when physical normal dispatch and persistent trace both diverged from the pinned CPU continuation. The first mismatch occurred at generated token index `10`. Production correctly retained windows `[2, 4, 8]`.

We need a bounded diagnostic path that identifies the first layer and value boundary where device execution diverges from the pinned FLA and official RWKV-LM recurrences. A fix must preserve all current production admission and cleanup contracts.

## What changes

- Add a device diagnostic that records bounded per-step and per-layer values for the known prompt through the first failing generated step.
- Extend the pinned CPU authority to emit matching comparison values.
- Classify the mismatch as a device implementation error, expected BF16 drift, or a CPU authority error with checked evidence.
- Add positive and negative regression tests for the diagnosed cause.
- If a source fix restores exact parity, create a new immutable window-16 candidate and repeat every admission gate.
- If exact parity cannot be restored within the bounded investigation, retain production windows `[2, 4, 8]` and archive the exact blocker.

## Non-goals

- This change does not weaken exact generated-token or FP32 recurrent-state parity.
- This change does not reuse the rejected candidate identity for admission.
- This change does not admit window `16` before a complete new evaluation.
- This change does not claim physical performance from simulator evidence.
