## Why

The local 0.74 runtime lacks the fused chunk GDN operation. The previous runtime-uplift archive selected no immutable candidate. The 0.77 release now provides a concrete source to evaluate.

Finding 3 in `docs/research/qwen38-p150x2-2026-09-04.md` records the source evidence.

## What Changes

- Build an isolated candidate from `tenstorrent/tt-metal@9f9cd4fd590f4b606bd0981a4fe0b6403eb38ec9`.
- Bind a compatible model tree, local patches, compiler, drivers, simulator, and Python dependencies.
- Evaluate fused GDN prefill and each currently disabled fusion separately.
- Retain the exact 0.74 control and every required fallback until matched evidence passes.

## Impact

Nix packaging owns the candidate closure. The Qwen admission core owns capability and performance decisions. The module default and global runtime pin remain unchanged.

API presence alone does not prove compatibility. Simulator evidence cannot establish hardware speed. A rejected candidate is a valid completed evaluation, not permission to weaken correctness gates.
