## Why

The traced P150x2 path proves a large single-token decode gain, but no local evidence shows that a GDN-only optimization is the best next investment. The checkpoint contains an unused MTP head, while the packaged runtime has no measured multi-token target-verification path.

## What Changes

- Add a diagnostic-only teacher-forced target-verification probe for widths 1, 2, and 4.
- Compare each width with matched sequential traced decode, including target-token and recurrent-state checks.
- Use a pure deterministic core and typed Nickel policy to reject incomplete or unstable evidence.
- Correct the pinned model architecture record from 56 linear and 8 full-attention layers to 48 linear and 16 full-attention layers.

## Impact

The package gains an explicit operator diagnostic and receipt contract. Production remains eager. The change does not load MTP weights, implement speculative acceptance, or admit an optimization.
