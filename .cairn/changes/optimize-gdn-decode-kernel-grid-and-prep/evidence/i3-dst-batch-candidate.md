# Dst batching: capacity correction

Recorded 2026-09-13. This record corrects the earlier geometry and acceptance claims.
The original candidate is rejected. The bounded replacement has no physical acceptance result.

## Proven source defect

The factory requires `key_dim = val_dim = 128` and uses `32x32` tiles.
Therefore `key_tiles = val_tiles = 4`, and `state_tiles = 16`.
The earlier record incorrectly used two value tiles and eight state tiles.

The factory requires FP32 destination accumulation and rejects full destination synchronization.
The default `ComputeConfig.dst_full_sync_en` is false.
TT-Metal therefore generates `DST_ACCUM_MODE = true` and `DST_SYNC_MODE = DstSync::SyncHalf`.

Blackhole has 1024 rows of sixteen 16-bit values in Dst.
Half synchronization and FP32 storage each halve that capacity.
A `32x32` tile occupies 64 rows in the LLK address model.
Thus the configured region holds `1024 / 2 / 2 / 64 = 4` tiles.
Valid destination indices are `0..3`, not `0..15`.

The original patch sends indices through 15 in four stages:
`state_decay`, `outer`, `outer_scaled`, and `state_new`.
The other batched stages use indices through three.
The host reproduction rejects the original installed kernel at the first invalid index.

### Source chain

- `pkgs/tt-metal/gdn-decode-fp32.patch`: head dimensions, precision validation, tile counts, and compute configuration.
- TT-Metal 0.77 source `tt_metal/api/tt-metalium/kernel_types.hpp`: the half-sync default.
- `tt_metal/jit_build/genfiles.cpp`: generated accumulation and synchronization macros.
- `tt_metal/hw/inc/internal/tt-1xx/blackhole/tensix_types.h`: Dst geometry.
- `tt_metal/tt-llk/tt_llk_blackhole/common/inc/ckernel.h`: `get_dest_max_tiles`.
- `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_math_binary_api.h`: the destination-index contract.
- Pinned ISA source `BlackholeA0/TensixTile/TensixCoprocessor/Dst.md` at `b7738d9ac14a34a4033d60dde9463466b23082e1`: physical Dst storage.

The source tree is `/nix/store/5cafvl5cvjzck5hjl36ixj48mpx6j28f-source`.
A separate read-only review reproduced the capacity calculation and found no applicable override.
This calculation is not a complete hazard audit.

## Bounded replacement

`pkgs/tt-metal/gdn-decode-dst-batch.patch` now obtains capacity from the pinned LLK helper.
Each state stage processes chunks of at most `dst_batch_tiles`.
Each chunk uses local destination indices and retains the original global CB tile indices.
The final partial chunk also stays within the state extent.

The value stages retain batching, with a compile-time assertion that their tile count fits Dst.
The transpose and `exp(g)` stages retain their original form.
The replacement does not change reader or writer kernels, precision, or the per-value reduction order.

This remains within-instance batching. It does not implement cross-instance pipelining or state splitting.

## Host regression evidence

`pkgs/qwen36/gdn-dst-capacity-check.cpp` executes the actual kernel body through a bounded symbolic API.
The fixture derives the FP32 half-sync capacity from package-matched headers.
It compares every stored tile expression with the per-tile control for one, two, and four instances.

- Original installed candidate: rejected with `Dst index exceeds configured capacity`.
- Bounded replacement: capacity and symbolic expression comparison pass.
- Negative control: a destination region smaller than the required value batch rejects.
- Compilation uses `-Wall -Wextra -Werror`.

The package check reverses only the Dst patch to recover its control kernel.
It does not depend on a sibling checkout or an old store output.
The fixture does not model SFPU numerics, concurrent TRISC execution, CB credits, or NoC ordering.

## Historical screens: not replacement evidence

The previous sessions reported these token-level comparisons:

| Round | Reference median | Original candidate median | Median change | Mean change |
| --- | ---: | ---: | ---: | ---: |
| 1 | 81.874 ms | 79.748 ms | -2.60% | -3.69% |
| 2 | 85.540 ms | 80.073 ms | -6.39% | -7.28% |

These were reference-then-candidate sessions, not the required kernel ABBA screen.
Their response fields compare tokens, text, state-slice labels, and reset status.
They do not compare recurrent tensor contents or prove that every destination index is valid.
The capacity defect invalidates the earlier inference that token parity proved tile placement.
No timing result transfers to the bounded replacement.

The original patch remains in `rejected-unbounded-dst-batch.patch` for reproduction.
The full kernel audits, representative simulator execution, and physical recurrence comparison remain open.
