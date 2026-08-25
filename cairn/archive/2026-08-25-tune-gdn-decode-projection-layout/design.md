## Context

The archived decode-component attribution measures the P150x2 decode step at
roughly `54 ms`, with GDN mixers near half of it and MLP most of the rest.
The GDN TP projections (`qkvz`, `ba`, `out`) call `ttnn.linear` with a
compute-kernel config but rely on automatic layout selection. The reviewed
Qwen3.6-A3B bundle faced the same shape class at batch one and measured
`AUTO 34.4 us -> DRAM-shard 17.8 -> wide-1D multicast 12.7` for a small-N
decode projection, and `1.71x` to `1.73x` for DRAM-sharded input
projections.

### Recorded mismatch: packaged decode projections are not AUTO (2026-08-25)

A device probe on the packaged qwen36 stack (`gdn-interior-split.v2`, B=1,
real weights, STRICT fabric) shows the decode TP projections already use
tuned 1D decode program configs. `Qwen36ModelArgs._init_tp_config` sets
`proj_1d_decode=True` and both `gdn_qkvz_decode_1d_progcfg` and
`gdn_out_decode_1d_progcfg`; `_project_qkvzab` (decode M=1) and `_row_proj`
route to `tpc.matmul_1d_decode`, not to AUTO `ttnn.linear`. `gdn_out_weight_memcfg`
is `None` (AUTO only there; qkvzab weights are DRAM-sharded). Phase split
medians (us): recurrence 3946 > all-reduce 2934 > qkvzab 2658 > out 1917,
rest 3372, step 14828 (isolated/serialized; down from the overlapped
attribution). No single op dominates; projections together are comparable to
the recurrence. Evidence: `.pi/evidence/qwen38/gdn-interior-split.json` and
`gdn-interior-split-FINDINGS.md`.

## Decisions

### Decision: Explicit program configs behind a default-off gate

**Choice:** Select wide-1D or DRAM-sharded program configs for the decode
projections only when the operator sets the candidate gate; prefill and the
default path stay on AUTO.

**Rationale:** The accepted spec keeps tuning non-admitted until matched
physical evidence exists. A default-off gate gives a clean eager baseline for
the required A/B pair without touching production behavior. The interior-split
probe (above) is the evidence floor that re-scopes which op the candidate may
target; the 1D-decode configs are the baseline to beat, not AUTO.

### Decision: Layout selection lives in a pure core

**Choice:** A pure helper maps `(K, N, device grid)` to the candidate program
config or to None when the shape does not qualify; the patch consumes only
that helper.

**Rationale:** Shape eligibility and tie-breaking become device-free testable
policy, matching the house core/shell split, and the adapter stays a thin
consumer.

### Decision: Evidence before any speedup claim

**Choice:** The candidate is selected, never admitted, by this change. Any
end-to-end claim must go through the existing decode-performance-admission
requirement with order-balanced physical roots.

**Rationale:** Per-op microbenchmarks do not imply model speedup; the spec
forbids that inference explicitly.

## Risks / Trade-offs

- Wrong shard choices can regress individual ops; the pure core bounds shapes
  it accepts and the gate defaults off.
- L1 pressure from sharded activations can collide with the L1-resident
  recurrence state; the first physical pair watches for allocation failures
  and treats them as a failed candidate.

### Implementation (2026-08-25, device-free)

The candidate is realized as `models/demos/blackhole/qwen36/tt/gdn/decode_layout_candidate.py`, a pure
module with no ttnn import, added by `gdn-decode-layout-candidate.patch`. The gate is
`QWEN35_GDN_DECODE_LAYOUT` (absent, empty, or non-listed values keep production). The pure
policy `select_decode_layout` accepts single-token decode, a tile-aligned K, a positive N
(N needs no tile alignment; the real 8240 width is legal), and a non-degenerate grid, and
returns the DRAM-sharded family. `gdn/tp.py` consumes it as the FIRST decode branch for the
qkvzab projection (DRAM-sharded `_col_proj` with `gdn_qkvzab_progcfg`), so the gated branch
wins over the default 1D-mcast path and prefill stays untouched. Out projection keeps the
1D-mcast path (its weight is interleaved; sharding it is out of scope for this candidate). The
new `check.py` probe (`probe_gdn_decode_layout_candidate`) exercises positive/negative gate and
policy cases and asserts branch ordering. On-device iteration found and fixed two real gaps:
(1) `sharded_decode_matmul` requires a WIDTH_SHARDED weight, which `proj_1d_decode=True` does
not load, so the candidate branch now converts and caches one sharded copy
(`_gdn_qkvzab_candidate_weight`); (2) the packaged binding exposes `memory_layout` as a
property, not a method. The matched physical A/B (`gdn-layout-ab.v1`, one layer, B=1, P150x2)
measure the candidate at qkvzab 416.5 us vs eager 364.7 us median (~14% slower on the op) with
step totals within noise (3099 vs 3188 us), so the candidate is NOT admitted; production keeps
the tuned 1D-mcast path and the gated capability stays as diagnostic tooling. Pre-build
device-free validation: 19 gate/policy assertions pass, the patched `tp.py` imports
device-free, the patch applies cleanly with `patch -p1 --dry-run`, and the full `qwen36-check`
suite passes all probes except one pre-existing stale-base import artifact (store package lacks
`mtp_trace_hygiene_core`), which the real build includes. The authoritative `nix build .#qwen36`
remains pending on a congested shared daemon; until it passes, the gate admission is not complete.
