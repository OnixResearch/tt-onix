## Why

Matched trace replay attributes 47.1% of each decode step to the 48 GDN mixers.
One GDN layer takes 594 microseconds against a 113-microsecond weight-read floor.
About 81% of the layer is a serial chain of small device operations.

The MLP path reaches about 70% of its weight-bandwidth floor.
The GDN path reaches about 19%.
The GDN decode chain is therefore the largest actionable efficiency loss.

## Outcome

Add a default-off GDN decode fusion candidate for the pinned P150x2 runtime.
The candidate folds fixed Q/K scaling into RMSNorm and runs the fp32 recurrence in one device program.
The unchanged path remains the default and the fail-closed fallback.

Select the candidate only after matched physical evidence passes fixed correctness and performance gates.
A rejected candidate must not change production.

## What Changes

- Add a pure, device-free closed-mode policy for `off`, `norm`, `state`, and `full`.
- Add a package-matched fp32 kernel that replaces the recurrent launch chain with one device program.
- Add positive and negative device-free checks for gate policy and source integration.
- Bind the patch and policy to the Qwen execution identity.
- Record matched physical token, state, reset, and timing evidence before any selection.

## Impact

- **Production:** unchanged unless new physical admission succeeds.
- **Runtime:** TT-Metal remains pinned at `0.74.0`.
- **Model:** Qwen3.8-27B revision remains unchanged.
- **State:** fp32 GDN recurrent state remains required.
- **Evidence:** any changed execution identity needs fresh B=4 admission.

## Out of Scope

- A TT-Metal pin uplift.
- BF16, TF32, or reduced-precision recurrent state.
- MTP or wider-batch admission.
- Service scheduler changes.
- Claims from isolated operation timing alone.
