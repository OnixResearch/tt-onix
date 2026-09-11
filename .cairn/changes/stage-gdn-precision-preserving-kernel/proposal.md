## Why

Phase 1 of this change inspected the pinned runtime and falsified the original
premise. This change was written to vendor an upstream fused GDN decode op.
That op already exists here:

- `pkgs/tt-metal/gdn-decode-fp32.patch` adds `ttnn.transformer.gated_delta_decode`;
- the pinned 0.77 runtime exports it (`gated_delta_decode` appears five times in
  `_ttnn.so`, `decode_gated_delta_rule` zero times);
- `pkgs/qwen36/gdn-decode-fusion-candidate.patch` wires it behind
  `QWEN35_GDN_DECODE_FUSION=full`;
- it is already admitted for sampled mode and for B=4, with a measured
  `1.0644x` full-step speedup.

There is nothing to vendor. The recorded falsification and the source facts
live in `evidence/upstream-authority.md`.

What remains is a narrower and harder problem. The fused route is rejected for
greedy single-stream serving on exact parity, and the archived root-cause search
established that the fused binary-SFPU route is **more** precise than the
composite, not less. The composite's mixed matmul kernel unpacks FP32 inputs to
TF32. The difference accumulates through 48 GDN layers and crosses late argmax
margins, with divergence at generated indices `47`, `32`, and `16`.

The archived search recorded the repair as blocked and named it exactly:
"Requires a staged kernel architecture that preserves each TTNN precision
boundary."

This change owns that repair. It is the only named route to the fused speed on
the greedy path, which is the path the serialized service actually serves.

## What Changes

- Replace the fused op's single-stage fp32 recurrence with a staged kernel
  architecture that reproduces each precision boundary the composite performs.
- Add a pure, device-free staging core that owns the stage order, the boundary
  list, and the closed validation.
- Add exact greedy parity evidence over a multi-prompt corpus, plus the existing
  GDN and full-step ratio bounds.
- Keep the current fused route reachable and default-off until the screen
  passes.

The change admits nothing until the screen passes and the owner reviews it.

## Impact

- **Files**: `pkgs/tt-metal/gdn-decode-fp32.patch` or a follow-up patch beside
  it, `pkgs/qwen36/` staging core and check probes, this change directory, and
  the physical evidence tree.
- **Testing**: qwen36 device-free check with positive and negative staging
  probes, package build, kernel hazard audits, a fresh Blackhole simulator
  smoke, and matched physical roots.
- **Production**: unchanged. No serving route moves.
