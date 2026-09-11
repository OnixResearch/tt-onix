## Why

GDN decode is the largest single block in the P150x2 decode step. The admitted
component profile measures 26.34 ms across 48 GDN layers against 56.33 ms per
token. That is 47 percent of the step, or 0.549 ms per layer.

The layer cost is not memory-bound. The 48-layer recurrent state costs 144 MiB
of read plus write per token, which is 0.29 ms at 512 GB/s. The cost is kernel
grid use and host preparation.

Upstream has already built the replacement kernels and measured them on this
exact 2x Blackhole p150a 1x2 mesh:

- `ttnn.transformer.decode_gated_delta_rule` (PR 53482 and PR 53587, open and
  unmerged) routes T=1 decode through a fused op. It measured 19.16 tok/s
  against a 17.75 tok/s unfused baseline on `traced_128`. Independent re-runs
  measured 19.12, 19.03, and 18.84 tok/s at 128, 4k, and 8k context.
- `ttnn.transformer.fused_recurrent_gated_delta_rule` (issue 50475, branch
  `atupe/qwen36-mtp-v2`) reports 0.345 ms per layer for decode, down from
  0.541 ms, and 2.3 ms per candidate for verify, down from about 18 ms.

The second kernel serves spec-decode verify at T=K+1. It is the missing piece
that made the MTP round uneconomic.

This repository tried a third path and it failed. The staged fusion candidate
`QWEN35_GDN_DECODE_FUSION=norm-state` measured a 0.9824 step ratio against a
0.97 target and stayed disabled. The upstream fused op measured a 0.9266 ratio
on the same mesh, so the op is materially stronger than the rejected local
candidate.

Neither upstream kernel is merged. Both must be vendored, gated, and measured
in this repository before any admission.

## What Changes

- Vendor the fused GDN decode and verify op into the Qwen package overlay
  against the pinned runtime, with a default-off gate.
- Add a pure, device-free selection core that rejects unknown gate values and
  unsupported shapes, and that selects the composite path otherwise.
- Preserve the composite decode graph byte-identically when the gate is off.
- Add a bounded physical screen: exact greedy token parity over a multi-prompt
  corpus, step-time ratio against the composite path, and recurrent-state
  parity against the composite reference.
- Re-run the rejected MTP drafter and rollback economics under the fused verify
  path, because the verify kernel is the named blocker for that rejection.
- Bind the op, its gate, and the selected mode into the package execution
  identity.

This change admits nothing on its own. The gate defaults off, and no serving
route changes until the physical screen passes and the owner reviews it.

## Impact

- **Files**: `pkgs/qwen36/` (new op overlay patch, gate policy core, check
  probes), `pkgs/qwen36/service/`, this change directory, and the physical
  evidence tree.
- **Testing**: qwen36 device-free check with positive and negative gate
  probes, package build, a fresh Blackhole simulator smoke, and matched
  physical roots in `control,candidate,candidate,control` order.
- **Production**: unchanged in this change. The gate defaults off and the
  composite path remains the served route.
