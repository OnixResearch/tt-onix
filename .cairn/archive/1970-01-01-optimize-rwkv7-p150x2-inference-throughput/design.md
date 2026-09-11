## Context

The target is RWKV7-Goose-World2.8-0.1B on two fabric-connected Blackhole P150A
cards. The model has twelve layers, hidden size 768, twelve heads, head size 64,
and a 382,111,072-byte BF16 checkpoint.

RWKV-7 replaces an expanding attention KV cache with fixed recurrent state. Each
head updates a 64 by 64 matrix with diagonal decay and rank-one terms. The complete
model state has 608,256 elements. It needs about 2.4 MB in FP32, while the weights
remain DRAM-resident.

The package-matched upstream CUDA kernel stores recurrent state as FP32 and accepts
BF16 operands. The local TT-NN runtime currently creates BF16 state tensors. Short
bounded parity does not prove that BF16 state remains stable over long contexts.

The current P150x2 runtime uses full-lockstep replication. Static source review also
finds about 1,230 wrapped TT-NN operation calls per token. This count is not a
measured kernel-launch count because some wrappers only change metadata.

## Goal and Success Contract

The goal is higher aggregate steady-state decode throughput for independent
requests. The measurement profile defines these named bounds:

- `minimum_dual_card_speedup = 1.50`
- `maximum_p99_latency_regression_fraction = 0.10`

The one-card run uses one active replica. The two-card run uses two active replicas.
Each replica receives the same number and shape of independent requests. Compile,
warm-up, evidence readback, and teardown time stay outside steady-state decode time.

The physical result passes only when both conditions hold:

```text
dual_aggregate_tokens_per_second / single_aggregate_tokens_per_second
    >= minimum_dual_card_speedup

(dual_p99_inter_token_latency - single_p99_inter_token_latency)
    / single_p99_inter_token_latency
    <= maximum_p99_latency_regression_fraction
```

The receipt also proves that each card generated distinct request IDs. A lockstep
result cannot satisfy this contract.

## Decisions

### Decision: Optimize weak-scaling throughput first

**Choice:** Run one independent complete model replica per card. Use continuous
batching inside each replica.

**Rationale:** The complete model fits easily in each card's DRAM. RWKV state is
small. Independent replicas avoid fabric collectives and let each card reuse weight
tiles across several active sequences.

Full tensor parallelism adds frequent collectives around hidden size 768. Layer
partitioning also adds a card boundary without overlap for one autoregressive
stream. Neither design is the weakest path to the throughput target.

### Decision: Keep fast mode separate from evidence mode

**Choice:** Preserve the existing evidence path. Add a fast path with no state
readback, profiler flush, control replay, or artifact write inside its timed decode
interval.

**Rationale:** Correctness evidence and performance measurement have different
contracts. Removing proof behavior from the old path would weaken existing
acceptance evidence.

### Decision: Use a functional scheduler core

**Choice:** A pure deterministic core maps ready request descriptors to card-local
replicas and batch slots. The imperative shell owns device queues, token input,
output collection, clocks, files, and receipt writes.

**Rationale:** State ownership and batching rules can be tested without hardware.
The shell remains a small TT-Metal orchestration layer.

The scheduler rejects duplicate state-slot ownership, unknown card IDs, batch
capacity overflow, and reuse of a live request ID.

### Decision: Keep recurrent state device-resident and FP32

**Choice:** Keep token-shift and WKV state on the selected card between decode
steps. Use FP32 for the recurrent update. BF16 checkpoint weights and ordinary
activations remain unchanged unless profiler evidence justifies another format.

**Rationale:** Host state transport adds synchronization. FP32 matches the upstream
recurrence contract and reduces long-context numerical risk.

Device-resident does not require every state tile to stay in L1. Initial work can
use device DRAM. L1 placement follows only when profiling identifies state traffic
as a material bottleneck.

### Decision: Optimize in profiler order

**Choice:** Apply these steps in order:

1. Remove evidence synchronization from the fast timed interval.
2. Add independent replicas and deterministic continuous batching.
3. Capture and replay the static decode graph when trace bindings are valid.
4. Read the physical operation and utilization profile.
5. Add a fused WKV operator only when launch or recurrence cost blocks the target.

**Rationale:** State traffic is small relative to the checkpoint weight stream.
Fusion has kernel correctness and synchronization risk. A measured decision avoids
premature custom kernel work.

If fusion is required, one operator fuses decay, both rank-one updates, recurrent
state write, and state projection. It accepts BF16 vectors, updates FP32 state, and
returns BF16 output. The implementation must pass the applicable LLK latency,
reconfiguration, MMIO, semaphore, dataflow, and performance audits.

### Decision: Use trace replay only after binding validation

**Choice:** Capture the static decode graph after weights load and warm-up. Replay
uses new token input and existing device state. If package-matched TT-Metal cannot
bind these inputs safely, retain normal dispatch and record the exact blocker.

**Rationale:** Trace replay can remove host launch gaps. An invalid trace that reuses
stale token or state addresses would silently corrupt generation.

### Decision: Run simulator smoke before silicon

**Choice:** Run a hardware-neutral `1x1` smoke with packaged `libttsim_bh.so` before
the next physical attempt. Require device open, firmware compilation, representative
weight load, one RWKV layer, and recurrent-state carry.

The same host and device binaries run on simulator and silicon. The simulator path
must not add runtime branches. The optional `libttsim_bh_x2.so` path remains a
preliminary mesh test only.

**Rationale:** A small simulator run can remove software and firmware compilation
risk before device ownership changes.

### Decision: Require physical profiler evidence

**Choice:** Bind the one-card baseline and two-card candidate to the same checkpoint,
runtime closure, profile, machine facts, prompt set, generation settings, and
sampling seed set. Record:

- Aggregate and per-card tokens per second.
- Mean, p50, and p99 inter-token latency.
- Compile and warm-up time outside steady-state results.
- Per-operation mean and p99 device latency.
- Program launch count and host-gap time.
- Math or ALU busy fraction and NoC idle fraction.
- Peak L1 and DRAM allocation per card.
- Per-card request IDs and generated-token counts.

The deterministic admission core computes both target equations. Missing or
non-finite metrics fail closed.

**Rationale:** Repository source and simulator output cannot establish physical
speedup.

## Validation Strategy

Device-free positive tests cover valid scheduler assignments, independent state
slots, deterministic profile export, passing metric fixtures, and FP32 state
metadata.

Device-free negative tests cover duplicate request ownership, lockstep request IDs,
batch overflow, BF16 recurrent state, missing card attribution, non-finite metrics,
and a false speedup claim.

Simulator validation covers one layer, representative weights, state mutation, and
state carry. It does not gate performance.

Physical validation first records the unchanged single-card and lockstep baselines.
After implementation, it records the one-replica and two-replica fast paths. Each
run uses the repeat distribution from the shared measurement profile.

Correctness validation compares each independent stream against the existing CPU
reference. It includes a short sequence and a named long-context sequence. The
long-context length belongs to the Nickel profile.

## Risks and Mitigations

- **Host scheduler saturation:** Record host gaps and per-card idle time. Increase
  queue depth only through profile values.
- **Low batch occupancy:** Record ready-sequence count and effective batch size.
- **Trace address reuse:** Add positive token-change and negative stale-binding
  fixtures before physical replay.
- **FP32 kernel pressure:** Keep state in device DRAM first. Use L1 only after a
  measured state-traffic finding.
- **Custom fusion races:** Run the complete applicable LLK hazard and performance
  audits before hardware use.
- **Simulator mismatch:** Mark every simulator receipt as simulated. Do not use it
  for placement or performance claims.
- **Target miss:** Preserve the exact profiler receipt and optimize only the largest
  attributed bottleneck. A bounded, measured miss is a valid intermediate result.

## Non-Goals

- Changing the accepted correctness tolerances.
- Replacing the existing evidence classifier.
- Claiming improved single-stream latency.
- Claiming ideal two-card scaling.
- Updating TT-Metal, checkpoint, tokenizer, or firmware pins.
