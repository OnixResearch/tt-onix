## Decisions

### Decision: traffic reduction is the candidate, not a demonstrated kernel

**Choice:** the candidate is **bank-matched DRAM-sharded weight residency** on
the MLP gate/up/down path across the 64 layer calls: width-shard weights across
exactly the 8 DRAM reader banks, bind the compute grid to that 1×8 shard grid,
and derive `per_core_N` from it. Dtype, equations, compute fidelity, accumulation,
collective order, the separate matmul plus `reduce_scatter_minimal_async`
contract, and every other projection's grid remain unchanged.

Plan a device-free fail-closed legality/rejection module behind a default-off
admission flag, with the numerical-retention screen defined before implementation.
Whether this layout reduces bytes or rereads relative to the packaged source
remains to be established. No implementation or measurement is claimed; no
variant is claimed to exist or win.

**Rationale:** the source-derived per-device one-read model is 89,128,960 B
payload / 97,484,800 B tiled storage per layer. At the separately traced 335.3 µs
MLP forward it yields 265.82 / 290.74 GB/s, or 51.9–56.8% of the packaged
512 GB/s P150a ceiling. Unchanged-bytes 2× would imply 531.6–581.5 GB/s;
that interpretation exceeds the ceiling and cannot justify admission ([B],
“Weight-byte model and the hardware ceiling” and “Sensitivity”; [C], appended
correction). These are modeled weight bytes, not DRAM counters or projection-only
measurements. Residency capacity and rereads remain to be characterized.

### Decision: retention and prior rejections gate implementation

**Choice:** P1 must fix the NLL/perplexity floor, corpus, and comparison method
before P2 implementation; P2 wires that screen into the chosen prototype. Any
variant must beat the already-rejected `QWEN36_MLP_DOWN_BF4` precision option
(rejected on NLL) and the `matmul_reduce_scatter_decode` decode-M=1 rejection,
not relabel either as an untried optimization.

**Rationale:** the precision rejection stands; at decode M=1 the fused
matmul/reduce-scatter collapses to roughly eight cores and loses ([A],
“Collective share” and “What this implies”; [C], “Already-rejected mechanisms”).
No numerical-retention threshold or successful candidate is assumed here.

### Decision: four gated phases

**Choice:** P1 is device-free packaged-source characterization of per-layer
rereads and residency, with no measurement. P2 is a device-free prototype of the
chosen traffic reduction, with the predefined retention screen wired. P3 is the
bounded hardware window named `mlp-projection-streaming-p3`, deferred until
physical devices 0 and 1 are released. Before that window, record its exact
commands, duration, baseline/candidate identities, retention comparison, and stop
conditions; no hardware command is authorized by this skeleton. P4 promotes only
under the existing admission rules.

**Rationale:** keep source evidence, arithmetic, and hardware measurements
separate. P1/P2 evidence must identify source revisions and exact locations,
record method, findings, residual assumptions, and coverage bounds, following
[D], “Method”, “Findings”, and its explicitly tagged inference style. No source
audit or prototype is represented as already complete.

## Risks / Trade-offs

- 0.3894 ms × 64 = 24.92 ms of the 48.14 ms component sum is broad MLP
  attribution, not recoverable projection time or an end-to-end step. The
  335.3 µs forward at widths 1 and 16 excludes norm/add but has no gate/up/down
  timing split ([A], “Component medians” and “Traced groups”; [B], “Step budget
  and timing boundaries”).
- Reducing modeled traffic may not improve execution; source-level residency
  feasibility and numerical retention are unresolved. No performance, latency,
  admission, or silicon claim; no new hardware, simulator, or measurement.

[A]: ../qwen38-tp2-collective-shape-tuning/evidence/step-attribution-2026-10-21.md
[B]: ../../../docs/research/qwen38-mlp-streaming-attribution.md
[C]: ../qwen38-tp2-collective-shape-tuning/evidence/ceiling-verdict-2026-10-22.md
[D]: ../optimize-gdn-decode-kernel-grid-and-prep/evidence/v4-kernel-hazard-audit.md

## P1 verdict, 2026-10-22

Under the conditional single-pass reader model, bank matching alone reduces
**no aggregate weight bytes or page reads**. The weight-reader count is
`Kt × per_core_N` for each assigned output-column partition, independent of
`in0_block_w`; repartitioning those columns among eight readers does not remove
any weight tile. Full-K participation is not evidence of full-matrix rereads
by every core. The source facts, conditional arithmetic, and coverage bounds
are recorded in [P1 layout evidence, section 4](evidence/p1-layout-current-2026-10-22.md#4-deeper-evidence-on-transactions-and-rereads-2026-10-22).

The unconditional per-step transaction delta is not established by any
permitted device-free authority. The actual selected-reader address/length
stream, duplicate multiplicities, edge handling, and candidate coalescing
have not been established. A congestion or routing improvement for unchanged
payload must not be relabeled as a demonstrated byte/reread reduction.

Therefore the candidate mechanism is **not supported device-free** and
**must not advance to P2 as written**. A successor must establish a mechanism
that changes aggregate bytes read or eliminates actual weight rereads, not
placement alone, while satisfying the unchanged numerical-retention and
separate matmul/reduce-scatter contract. The requirement and `specs/` remain
intact as the contract for any successor; no prototype or hardware probe is
authorized by this verdict.
