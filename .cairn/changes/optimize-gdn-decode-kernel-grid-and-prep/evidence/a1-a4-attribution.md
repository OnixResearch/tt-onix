# A1-A4: GDN decode occupancy, cost, host preparation, and a correction to the expectation

Recorded 2026-09-12 for change `optimize-gdn-decode-kernel-grid-and-prep`.

## A1: the admitted GDN decode cost, instances, and derived occupancy at TP=2

Sources: the B=1 component profile
(`docs/evidence/tt-metal-0.77/qwen-readback/component-profile.json`, summarised in
`docs/qwen-readback-evaluation.md`) and the matched admission receipts.

| Measurement | Value |
| --- | ---: |
| Full trace replay | 54.34 ms |
| GDN mixer, weighted over 48 layers | 26.34 ms (48.5 percent of the step) |
| GDN per layer | 0.549 ms |
| MLP, weighted over 64 layers | 23.86 ms |
| Full-attention mixer (16 layers) | 5.65 ms |
| LM head and argmax | 2.07 ms |

The fused decode factory maps one instance per `(batch, value head)`. At TP=2 a
device owns 24 value heads, so instances per launch are `24 x B` against 110
measured compute cores:

| Batch | Instances | Occupancy `min(instances,110)/110` | Instances per core |
| ---: | ---: | ---: | ---: |
| 1 | 24 | 21.8 percent | 1 |
| 4 | 96 | 87.3 percent | 1 |
| 8 | 192 | 100 percent | 2 |
| 16 | 384 | 100 percent | 4 |

So the admitted decode is **idle at B=1** (24 of 110 cores) and **looping at
B>=8** (2 and 4 sequential instances per core). Those are two different problems,
and the plan's single mechanism addresses only the first.

## A2: reconciliation with the upstream figures

The upstream report (issue 50475, and the second hardening item in PR 53482)
describes the fused decode as opening one instance per `(batch, value head)` and
using about 7 percent of the grid at TP=4, where a device owns 12 value heads and
the admitted path runs 8 instances.

Locally:

- the step share agrees closely: 48.5 percent of the replay here against the plan's
  47 percent figure;
- the occupancy does not: **21.8 percent at B=1 and TP=2, not 7 percent**, because
  TP=2 doubles the local value-head count to 24 (and B=4 already reaches 87.3
  percent);
- the per-layer cost here is 0.549 ms, consistent with the plan's own figure.

The mismatch is structural, not a measurement disagreement: the upstream
occupancy argument is a TP=4 argument. At TP=2 the same kernel is already near
full grid use from B=4 upward.

## A3: host preparation ops in the admitted decode path

From the applied fused-recurrence call site
(`models/experimental/gated_attention_gated_deltanet/tt/ttnn_delta_rule_ops.py`,
the `fused_recurrence` branch), the admitted path performs before the kernel
launch, per step per layer:

| Op | Count | Purpose |
| --- | ---: | --- |
| `ttnn.reshape` | 5 | flatten each of q, k, v, beta, g to head-major `[B*H, 1, D]` |
| `ttnn.permute` | 6 | head-major order for q, k, v, beta, g, and the output |
| `ttnn.unsqueeze` | 2 | beta and g to `[B, H, 1]` |
| `ttnn.reshape` (state) | 2 | state and state output to `[B*H, K, V]` |
| `ttnn.typecast` | up to 1 | state to fp32 when it is not already |

That is roughly 15 host-issued ops per layer per step, on top of the GDN kernel
itself. The profile's "host input update" subtotal is 0.470 ms for the whole
step, so the visible host chain is small in aggregate; the plan's item is to fold
it anyway, which removes launch cost rather than measured time.

## A4: state bandwidth is not the binding constraint

Per head the recurrent state is `128 x 128 x 4 B = 64 KiB`; per device per layer
that is `24 x 64 KiB = 1.5 MiB`, and across 48 layers `72 MiB` per step, so
read plus write is about `144 MiB` per step. Over the 54.34 ms replay that is
about **2.7 GiB/s**, and the state is held one head per core, i.e. 64 KiB in each
core's L1 rather than a DRAM stream. Neither figure approaches a bandwidth limit,
and the component evaluator's own numbers put the cost in the mixer, not in
state movement. Bandwidth is therefore not binding, as the task asks to confirm.

## The correction the attribution forces

**Splitting the state across cores per head cannot improve B>=8, and its benefit
at B<=4 is bounded by a fixed per-instance cost.**

Per-core work is `(instances / cores) x work_per_instance`. A value-block split
by `NV` makes `instances` larger by `NV` and `work_per_instance` smaller by `NV`,
so the *variable* part cancels exactly: at B=8 or B=16, where instances already
exceed cores, splitting leaves per-core work unchanged. It helps only where cores
are idle (B=1 at 21.8 percent, B=4 at 87.3 percent).

Worse, the per-instance cost is not fully divisible: the kernel transposes `k` to
`k_col` (`K x K`, independent of the value width) and pays reader, CB, and
launch overhead per instance. Splitting multiplies that fixed part by `NV`, so at
B=1 the state split buys occupancy at the price of `NV` times the fixed cost per
head. Whether it wins is a measurement, and the win is capped:

| Batch | Instances | Split benefit available | Reason |
| ---: | ---: | --- | --- |
| 1 | 24 | up to ~4x on the variable part | 86 idle cores |
| 4 | 96 | small | 14 idle cores |
| 8 | 192 | none | already 2 per core |
| 16 | 384 | none | already 4 per core |

The measured retention loss tracks this exactly: retention is 0.88 at one
instance per core, 0.735 at two, and 0.634 at four (see
`evidence/retention-scaling.md`). So the retention floor that blocks both B=8 and
B=16 needs the **per-core instance loop** addressed — by pipelining instances
within a core to hide the per-instance latency (the plan's I3), or by reducing
per-instance cost — and not by the state split (I2), which is a B<=4 occupancy
measure.

This does not reject the plan; it reorders it. I2 remains the right change for the
low-occupancy case the upstream report describes, and I3 is the mechanism that can
move the retention bound.

## Non-claims

- Occupancy is derived from instance counts and the measured 110-core grid, not
  from a device occupancy counter.
- The component profile is a B=1 measurement; the B>=8 statements are derived from
  instance arithmetic and the retention-scaling evidence, not from a per-component
  profile at width.
- The per-instance fixed cost is argued from the kernel's transpose and reader
  structure, not measured separately.
