## Why

Single-stream decode is weight-bandwidth bound. The admitted P150x2 step takes
56.33 ms for a 27.49 GiB model split across two cards at 512 GB/s each. That
places the weight-bound floor at 28.8 ms, or 34.7 tokens/second. The service
reaches 17.75 tokens/second, which is 51 percent of that floor.

That ceiling is set by bytes, not by kernels. Every weight family that moves
from BFP8 to BF4 removes half of its bytes from the per-token read, and the
ceiling moves up with it.

Upstream PR 53543 already registers three opt-in BF4 cache families for this
checkpoint and stores each under a distinct `.bfp4` cache name so the default
BFP8 bins are never overwritten:

- `QWEN36_LM_HEAD_BF4`
- `QWEN36_MLP_DOWN_BF4`
- `QWEN36_GDN_BF4`

The repository already carries the gates, and the accepted specification
already holds BFP8 as production precision until one named candidate has
complete numerical and matched physical evidence. This change supplies that
evidence path.

Precision is not free. The published Qwen3.8 MTP acceptance rate falls from
92.2 percent at BF16 to 84.8 percent at FP8, so a precision change can move
quality in ways a token-parity check on one prompt would miss. The evidence
must therefore include a bounded accuracy floor, not only parity.

## What Changes

- Exercise each opt-in BF4 cache family separately and together under distinct
  `.bfp4` cache names, and prove the default BFP8 caches are never overwritten.
- Add a pure, device-free cache-name and gate policy core with closed
  validation.
- Add numerical evidence per family: exact greedy token and text parity on a
  fixed corpus, plus a bounded accuracy floor that the profile owns.
- Add matched physical evidence: a decode step-time ratio per family that
  attributes the byte reduction, on devices 0 and 1.
- Keep the accepted `low_precision_non_admission` requirement in force. This
  change adds the admission contract beside it and does not weaken it.

This change admits no family by itself. Each family moves to production only on
its own passing evidence and owner review.

## Impact

- **Files**: `pkgs/qwen36/` (cache-name policy core, check probes, weight-cache
  generation paths), this change directory, and the physical evidence tree.
- **Testing**: qwen36 device-free check with positive and negative cache-name
  probes, package build, a fresh Blackhole simulator smoke, numerical screens
  per family, and matched physical roots.
- **Production**: unchanged. BFP8 stays the production precision until a named
  family passes.
