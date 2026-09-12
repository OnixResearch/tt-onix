# Prefill SCAN chunking: exact wrap points

Recorded 2026-09-11 for change `lift-gdn-grid-instance-cap`, task I3.

The decode factory is chunked (commit `8bf9f35`). The prefill guard is the
second site, in `distribute_scan` in
`chunk_gated_delta_rule/device/chunk_gdn_phased_program_factory.cpp`:

```c
const uint32_t ncores = grid.x * grid.y;
TT_FATAL(BH <= ncores, "num_heads {} exceeds compute cores {}", BH, ncores);
```

and a second cap in the `NV` search at `BH * cand <= ncores`.

## Current SCAN runtime arguments

```c
reader.emplace_runtime_args(core, {h, vb, NC, vb_buf, kd_buf, qd_buf, it_buf, kdec_buf, dl_buf, ti_buf, s0_buf});
writer.emplace_runtime_args(core, {h, vb, NC, o_buf, fs_buf});
compute.emplace_runtime_args(core, {NC});
```

One `(head, v-block)` pair per core, with `NC` as the inner sequential loop
over chunks.

## Proposed argument layout

```c
reader.emplace_runtime_args(core, {wi_start, wi_count, NV, NC, vb_buf, ...});
writer.emplace_runtime_args(core, {wi_start, wi_count, NV, NC, o_buf, fs_buf});
compute.emplace_runtime_args(core, {wi_count, NC});
```

Each kernel derives `h = wi / NV` and `vb = wi % NV` per instance, matching the
PREP phase's `wi`-flat convention. `NV` is already known at build time in
`ScanWorkDist::NV` and is uniform across cores, so it fits a runtime argument.

## Exact wrap points, from source

### `kernels/dataflow/reader_chunk_gdn_scan.cpp` (103 lines)

Arguments 0..10 today: `h, vb, NC, vb_addr, kd_addr, qd_addr, it_addr, kc_addr,
dl_addr, ti_addr, s0_addr`.

**Head-independent, stays outside the loop:** `tb`, every `TensorAccessor`
construction, the `constexpr cc/ck/kc`, `Noc noc`, and both lambdas
(`read_into`, `read_vslice`). The lambdas capture by reference and take their
row base as a parameter.

**Head-dependent, must go inside the loop:** exactly two statements.

```c
read_vslice(s0_acc, cb_S, h * Kt * Vt_full, Kt);   // the ONLY pre-loop use of h
for (uint32_t c = 0; c < NC; c++) { ... }          // reads to end of file
```

So the instance loop body is the initial-state read plus the chunk loop. This is
a small wrap, not a restructure.

### `kernels/dataflow/writer_chunk_gdn_scan.cpp`

Arguments 0..4 today: `h, vb, NC, o_addr, fs_addr`.

Body order: the `for (c < NC)` output loop first (from line 46), then a separate
state write loop (from line 65). **Both** must sit inside the instance loop,
because the state write is per-head.

### `kernels/compute/chunk_gdn_scan.cpp`

`const uint32_t NC = get_arg_val<uint32_t>(0);` at line 112, then
`for (uint32_t c = 0; c < NC; c++)` at line 122. Compile-time arguments
`Ct, Kt, Vt` stay untouched. The instance loop wraps the chunk loop; the compute
kernel gains `wi_count` as its first runtime argument and `NC` becomes second.

## Why this is the harder of the two sites

The decode op had one flat head index and no value blocking. SCAN has an inner
`NC` chunk loop and an outer `(head, v-block)` fan-out, so the instance index is
two-dimensional and `NV` must reach the kernels. The wrap points above keep the
change to an added loop plus two derived indices per kernel.

## Non-claims

This is a source reading, not a measurement. No mechanism is claimed to work and
no build has been attempted for this site.
