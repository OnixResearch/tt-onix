# D1: mechanism results, arithmetic, and non-claims

Recorded 2026-09-11 for change `lift-gdn-grid-instance-cap`, task D1.

## Both mechanisms

**Decode (`gated_delta_decode`, fused fp32 recurrence).** The
`num_heads <= compute_grid.x * grid_y` guard is gone; each active core takes
`ceil(num_heads / ncores)` contiguous head-instances with `(head_start,
head_count)` passed as runtime arguments. Verified: the guard fatal
reproduced device-free on the shared runtime and absent on the variant; the
chunked factory accepts the 192-instance launch; 192 instances are bitwise
equal to eight serialized 24-instance calls on output and recurrent state on
both physical P150s; the served B=8 path returns text-exact results at width
8 while the shared runtime fails the same group.

**Prefill (`chunk_gdn_phased`, `distribute_scan`).** The `BH <= ncores` fatal
is gone; the NV v-blocking search keeps its original bound; each active core
takes a contiguous chunk of `(head, v-block)` instances with `(wi_start,
wi_count)` as runtime arguments and `NV` passed so each kernel derives
`h = wi / NV`, `vb = wi % NV`. Verified: 192 instances bitwise equal per
position and in final state to the serialized per-user reference on both
physical P150s at two sequence lengths.

## Chunk arithmetic

Measured physical grid: **11 x 10 = 110 compute cores** on both P150s (the
device-free probe's planning constant was exact; the simulator's 13 x 10 = 130
is not the harvested physical grid).

| Batch | Instances (24xB) | chunk length | active cores | site behavior |
| ---: | ---: | ---: | ---: | --- |
| 1 | 24 | 1 | 24 | in-range, mapping identical |
| 4 | 96 | 1 | 96 | in-range, mapping identical |
| 8 | 192 | 2 | 96 | overflow, chunked decode and scan |
| 16 | 384 | 4 | 96 | overflow, chunk length 4 |

Decode: `heads_per_core = ceil(num_heads / ncores)`; `head_start = i *
heads_per_core`. Scan: `NV` from the unchanged bounded search (`BH * cand <=
ncores`), so at BH=192 no candidate above 1 qualifies and `NV = 1`; `per_core
= ceil(BH*NV / ncores)`; `wi_start = i * per_core`. Both use uniform ceiling
chunks, so the in-range case (`n <= ncores` or `total <= ncores`) collapses to
`chunk = 1`, `active = n`, and core `i` serving instance `i` — the pre-change
mapping exactly, which V3 proves over the whole in-range domain.

## Parity tables

Decode, direct op (bitwise, `torch.equal`):

| Device | Instances | Output comparisons | State comparisons | Result |
| --- | ---: | ---: | ---: | --- |
| 0 | 192 vs 8x24 | 8 | 8 | all exact |
| 1 | 192 vs 8x24 | 8 | 8 | all exact |

Scan, op level (bitwise): 2 devices x 2 sequence lengths x 8 users x
(output, state) = 64 exact comparisons, all equal.

Served B=8 (matched roots `single,batch,batch,single`, 3 measured runs per
prompt over 8 prompts):

| Root | Responses | Text-exact vs single-1 | Widths |
| --- | ---: | ---: | --- |
| single-1 | 32 | reference | 1 |
| batch-1 | 32 | 24/24 | 8 |
| batch-2 | 33 | 24/24 | 8 (one width-1 group) |
| single-2 | 32 | 24/24 | 1 |

## Non-claims

- **No admission moves.** No `.ncl` policy, profile, or binding file is
  modified by this change; the B=4 admission, the B=8
  `physically_admitted = false` binding, the trace-reuse and kernel
  distribution profiles, and the MTP bindings are untouched.
- **No timing claim.** Loop-level and served timings are recorded as
  diagnostics only and were computed outside the profile's receipt builder.
- **Served prefill is still per user.** The batched-scan parity is an
  op-level mechanism result; no served batched-prefill path was added or
  claimed.
- **No physical-placement, performance, or profiling claim.**
- **Simulator evidence is marked simulated**; it proves launch-level behavior
  only. TTsim cannot execute the fused op to completion at any width
  (unmodeled `ALU_ACC_CTRL_Zero_Flag_disabled_src`), which is why parity is a
  silicon result.
- **Single reader artifact recorded, not fixed**: the decode reader's pre-loop
  `reserve_back` block is dead weight (the API call is a stateless space wait).
  Removing it is deferred to keep the patch stable through verification.
- **One pre-existing defect noted, not owned**: `training-platform` fails with
  a byte-identical derivation before and after this change, and the
  `repository-quality-artifacts` manifest is invalid JSON at `origin/main`.

## Files changed

Eight evidence files plus this change's `tasks.md`,
`flake-modules/onix-packages.nix` (variant `gdnGridRuntime = true` and the
stale `qwen36` binding repair), `pkgs/qwen36/check.py` (the in-range mapping
probe), and `pkgs/qwen36/default.nix` (the two-directional kernel-content
binding). No receipt, profile, policy, or binding artifact changed.
