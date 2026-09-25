## Context

In decode, a GDN layer runs the `[qkv|z|a|b]` projection, then about 60 small
TTNN operations, then the out projection. The fused recurrence (`full`) replaces
the composite recurrence with one kernel. It keeps the chain around that
kernel: conv shift and multiply-accumulate, slices, reshapes,
`repeat_interleave`, permutes, and typecasts. Measured per-op costs are in
`evidence/attribution.md`. At B=1, dispatch and small data-movement programs
dominate that chain, not arithmetic.

## Decisions

### Decision: Two custom programs through `ttnn.generic_op`

**Choice:** `conv_prep` and `recurrence` are program descriptors built in
Python. Their kernels are JIT-compiled from `tt/gdn/kernels/` in the model tree.

**Rationale:** A TTNN C++ operation would require a multi-hour runtime rebuild
for each kernel iteration. It would also change the shared runtime identity.
`generic_op` keeps the programs in the model package and hashes them by kernel
source and compile arguments. Runtime arguments carry buffer addresses, so
trace replay works unchanged.

### Decision: Split at the conv state

**Choice:** The conv runs in its own program. Each core owns a disjoint set of
channel tiles and performs the state shift for its channels. The recurrence reads
the fp32 conv output.

**Rationale:** Three value heads share each key head's q and k channels. One
fused program would need a cross-core barrier before any core shifts those
channels' state. Two programs order the shift after every read at the cost of one
launch.

### Decision: Row-0 vectors in zero-padded tiles

**Choice:** For instance `(b, h)`, the reader copies batch row `b` of the q, k, v,
and gate tiles into row 0 of CB pages. Those pages are zeroed once per launch,
and the reader places the a/b/`dt_bias`/`-exp(A)` scalars at element (0, 0). The
writer stores row 0 of the bf16 result into batch row `b` of a persistent buffer
that is zero-initialized and shared by all GDN layers.

**Rationale:** The zero padding keeps the rank-1 update `k^T delta` exact. The
same code serves every batch width up to one tile row. It also removes the
GQA repeat and head-major relayout: the reader indexes key head `h / (Nv / Nk)`
directly. The B-row view of the shared buffer is a zero-copy reshape with the
padded shape unchanged.

### Decision: Exact fp32 recurrent state

**Choice:** The recurrent state, the decay factor, `v`, `v_read`, beta, and the
gate inputs use `UnpackToDestFp32` CBs. The carried state is decayed with an fp32
SFPU multiply. The rank-1 update accumulates into that DEST tile. Only matmul
operands are tf32.

**Rationale:** The FPU path truncates the carried state to tf32 on every step,
which acts as a systematic extra decay. `evidence/kernel-numerics.md` records
both variants.

### Decision: Greedy routing follows the startup mode

**Choice:** The router derives the deterministic route from the GDN layers'
startup mode. `step` keeps greedy requests on `step`. Every other mode keeps the
composite exact-parity route. Sampled requests are unchanged.

**Rationale:** One source of truth, the environment gate read by the layers,
prevents the router from switching a step-mode service back to the composite
route on its first greedy request.

## Risks / Trade-offs

- Greedy tokens differ from the composite route. Owner-approved on
  2026-09-24 under the quality gate. `evidence/model-eval.md` shows no NLL increase
  and smaller logit distance than the admitted fused recurrence.
- The `recurrence` program uses one core per head at B=1, or 24 of 110 cores. Its
  48 µs is the next target within the layer.
- MTP speculation has no evidence on this route, so the module rejects the
  combination.
