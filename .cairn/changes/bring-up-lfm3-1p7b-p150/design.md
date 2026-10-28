## Context

LFM3-1.7B has three operator families in one stack. The convolution layers hold a
seven-row causal window per channel, the sliding layers cap attention at 256
keys, and the global layers rotate only the high-frequency half of each head.
The checkpoint's `modeling_lfm3.py` is the authority for the exact ordering of
the per-head gate, the two sandwich norms, and the partial-RoPE tail.

Two numeric facts shape the design. First, TT-Metal's attention kernels accept
bfloat16-family inputs only, so a wide residual stream cannot use the fused
prefill SDPA. Second, the packaged matmul reaches about 1e-3 relative on
float32 inputs and no fidelity setting improves it (measured in
`evidence/bring-up-2026-10-27.md`), so a strict per-element oracle gate is
unattainable and the acceptance criteria are stated on the token stream plus a
reported per-layer deviation.

## Decisions

### Decision: Repo-authored TT-NN runner instead of a tt_transformers path

**Choice:** Implement the model directly in `pkgs/lfm3/runner` with TT-NN ops,
using the packaged runtime and the shared Metalium overlay.

**Rationale:** The architecture has no generic HF walker: the convolution
recurrence and the per-head gate have no `tt_transformers` implementation. A
custom TT-NN module is smaller than teaching the generic path three new block
types, and it keeps the device code in LGPL-2.1-only repository code instead of
an upstream patch.

### Decision: The float32 mode runs attention explicitly

**Choice:** With `--dtype float32`, prefill attends with `ttnn.matmul` on float32
q/k/v, a float32 softmax, and a host-built additive mask applied in 256-row
query blocks. Decode keeps the bf16 SDPA kernel by default and offers
`--decode-attention fp32` for a float32 cache and explicit float32 attention.

**Rationale:** Casting the stream to bf16 for SDPA throws away the accuracy that
the wide stream buys. The explicit path costs three matmuls and a mask add per
block and needs a block size so the score tensor stays bounded at the 4096-token
cache length. At 430+32 tokens the wide decode reproduced the default's 32/32
tokens at 4.5x the per-token cost, so the fast kernel stays the default and the
wide path is the invariance option.

### Decision: Fuse projections, split on the device

**Choice:** One `[q | k | v]` matmul per attention layer and one `[w1 | w3]`
matmul per FFN, split with `ttnn.slice`. The per-head gate stays its own matmul.

**Rationale:** Fusion removes 36 dispatches per token at the decode shapes and
measured cheaper per matmul (`2048->5120` at 0.057 ms against three separate
2048-wide matmuls at 0.14 ms). The gate cannot join the QKV matmul: its weight
is `[16, 2048]`, so the fused output width would be 3088 and miss the tile
alignment the split needs.

### Decision: The recurrent window is six row-major device rows

**Choice:** Decode keeps the last six pre-conv rows as separate `[1, 1, C]`
row-major tensors and computes the convolution as one `concat` + `multiply` +
`sum` over the window and the new row.

**Rationale:** Row-major slicing has no tile-alignment constraint, so the window
can be built without host round trips; three dispatches replace twelve
multiply/add pairs. Prefill accumulates the seven shifted taps on the whole
sequence instead, which is one pass per tap over the row-major operand.

### Decision: The oracle stays outside the derivation

**Choice:** `pkgs/lfm3/oracle/reference.py` is a repository script, not a package
entry point.

**Rationale:** The checkpoint declares `transformers_version = 5.12.1` and
imports `transformers.integrations.use_kernel_func_from_hub_with_fallback`,
which nixpkgs does not provide. Shipping a broken entry point or a second
transformers closure would both be worse than documenting the matching
virtualenv. The device gate consumes the npz the oracle writes.
