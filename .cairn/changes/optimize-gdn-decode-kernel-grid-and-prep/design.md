## Context

The upstream fused GDN decode kernel reports 0.345 ms per layer against 0.541 ms
for the composite. On this hardware the composite path measures 0.549 ms per
layer, so the local baseline agrees with the upstream composite figure.

The grid shape explains why the layer is slow. One program instance covers one
`(batch, value head)` pair. At TP=2 the fused op reports `B=1 H=24 K=128 V=128`,
so a single-token step opens 24 instances. The decode grid holds roughly 110
compute cores on Blackhole. Occupancy is therefore about 22 percent at TP=2 and
about 7 percent at TP=4.

A layer that uses a fifth of the grid and still costs 0.549 ms is not limited by
math throughput. It is limited by how little work is spread across the device
and by how many separate ops surround it.

The recurrent state is 2 MiB per layer per device at `H=24, K=V=128, fp32`. Over
48 layers that is 144 MiB of read plus write per token, which is 0.29 ms at
512 GB/s. State bandwidth is therefore not the binding constraint, and moving
the state on chip would not repay the work. The binding constraints are grid
occupancy and the host op chain.

## Decisions

### Decision: Split the state across cores before pipelining heads

**Choice:** First split the 128x128 state per value head across several cores.
Add head pipelining only where the state split alone leaves occupancy low.

**Rationale:** The state split raises parallelism without changing the recurrence
order inside a head. Head pipelining changes the execution structure more and is
harder to prove parity for. The upstream author lists both, and the cheaper
option is tried first.

### Decision: Fold host preparation into the kernel rather than batching it

**Choice:** Move the five fp32 typecasts, the two L2 norms, and the query scale
into the reader or compute kernel. Do not keep them as separate device ops.

**Rationale:** Batching separate ops would reduce launch count but keep the
intermediate traffic. Folding removes both the launches and the intermediates.
The upstream author names folding as the intended fix.

### Decision: Require exact recurrence parity, not a PCC bound

**Choice:** The screen compares the recurrent state and the decode output against
the admitted reference and requires exact parity over the corpus.

**Rationale:** This repository already rejected a GDN fusion candidate whose
state PCC stayed near 0.99997 over two steps but drifted enough to flip late
argmaxes across 48 layers. An approximate bound does not describe that hazard.

### Decision: Keep the state in fp32 and the verify path out of scope

**Choice:** Do not evaluate bf16 or bf8 inputs in this change. Do not touch the
T=K+1 verify path.

**Rationale:** A numeric-type change needs its own accuracy gate and its own
owner review. The verify path belongs to the fused-op change. Keeping both out
keeps each result attributable.

### Decision: Gate on occupancy and step time after parity

**Choice:** Read the occupancy floor and the step-time bound only after parity
passes. Reject on unstable duplicate roots as the existing discipline requires.

**Rationale:** A faster kernel with a different recurrence is not an
optimization. Parity first keeps the timing numbers meaningful.

## Risks / Trade-offs

- Splitting the state may add reduction traffic that cancels the occupancy gain.
  The step-time bound decides this, and a null result is a valid outcome.
- Folding the norms into the reader changes the reader's numeric order. Parity
  over a multi-prompt corpus is the guard.
- The upstream headroom list is an author's estimate, not a measurement. Every
  item must be re-measured locally before it is believed.
- Occupancy is a device-level quantity that the host cannot fully observe. The
  receipt must state how occupancy was derived and must not overstate it.
