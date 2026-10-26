## Decisions

### Decision: traffic reduction is the candidate, not a demonstrated kernel

**Choice:** the candidate is **bank-matched DRAM-sharded weight residency** on
the MLP gate/up/down path across the 64 layer calls: width-shard weights across
exactly the 8 DRAM reader banks, bind the compute grid to that 1×8 shard grid,
and derive `per_core_N` from it. Dtype, equations, compute fidelity, accumulation,
collective order, the separate matmul plus `reduce_scatter_minimal_async`
contract, and every other projection's grid remain unchanged.

P1 evaluated that proposal and **rejected bank matching as a traffic-reduction
mechanism**. The exact selected reader already requests every weight tile once;
placement alone removes no aggregate matrix bytes or weight rereads. No
fail-closed prototype, admission flag, or successor was implemented. P2 remains
blocked; the completed negative source audit and bounded retention contract are
recorded in the 2026-10-25 verdict below.

**Rationale:** the source-derived per-device one-read model is 89,128,960 B
payload / 97,484,800 B tiled storage per layer. At the separately traced 335.3 µs
MLP forward it yields 265.82 / 290.74 GB/s, or 51.9–56.8% of the packaged
512 GB/s P150a ceiling. Unchanged-bytes 2× would imply 531.6–581.5 GB/s;
that interpretation exceeds the ceiling and cannot justify admission ([B],
“Weight-byte model and the hardware ceiling” and “Sensitivity”; [C], appended
correction). The exact selected-reader enumeration now establishes that same
weight-byte total as software-requested bytes, not DRAM-controller counters or
projection-only measurements. GDDR ownership and bounded L1 CB residency are
characterized in the new reader evidence.

### Decision: retention and prior rejections gate implementation

**Choice:** P1 must fix the NLL/perplexity floor, corpus, and comparison method
before P2 implementation; P2 wires that screen into the chosen prototype. Any
variant must beat the already-rejected `QWEN36_MLP_DOWN_BF4` precision option
(rejected on NLL) and the `matmul_reduce_scatter_decode` decode-M=1 rejection,
not relabel either as an untried optimization.

**Rationale:** the precision rejection stands; at decode M=1 the fused
matmul/reduce-scatter collapses to roughly eight cores and loses ([A],
“Collective share” and “What this implies”; [C], “Already-rejected mechanisms”).
The original six-passage B1 diagnostic is now fixed at matched mean delta NLL
≤ +0.010 nats; no global quality floor or successful bank-matching treatment is
claimed. See the [retention screen](evidence/p1-retention-screen-2026-10-25.md).

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
[D], “Method”, “Findings”, and its explicitly tagged inference style. P1 is now
complete as a negative source characterization and rejection closure; no P2
prototype exists or is authorized by that result.

## Risks / Trade-offs

- 0.3894 ms × 64 = 24.92 ms of the 48.14 ms component sum is broad MLP
  attribution, not recoverable projection time or an end-to-end step. The
  335.3 µs forward at widths 1 and 16 excludes norm/add but has no gate/up/down
  timing split ([A], “Component medians” and “Traced groups”; [B], “Step budget
  and timing boundaries”).
- Lower modeled traffic alone would not establish faster execution. The current
  reader has no removable weight rereads; alternative persistent-cache legality
  is not established. Retention is bounded to the original six-passage B1
  diagnostic. No performance, latency, admission, or silicon claim; no new
  hardware, simulator, or timing measurement.

[A]: ../qwen38-tp2-collective-shape-tuning/evidence/step-attribution-2026-10-21.md
[B]: ../../../docs/research/qwen38-mlp-streaming-attribution.md
[C]: ../qwen38-tp2-collective-shape-tuning/evidence/ceiling-verdict-2026-10-22.md
[D]: ../optimize-gdn-decode-kernel-grid-and-prep/evidence/v4-kernel-hazard-audit.md

## P1 verdict, 2026-10-25

The [exact packaged-reader audit](evidence/p1-reader-transactions-2026-10-25.md)
supersedes the unresolved selected-reader questions in the historical
[2026-10-22 layout evidence](evidence/p1-layout-current-2026-10-22.md).
The installed configuration builder and a compiled extraction of the selected
0.77 reader address/edge loop were exercised for all six B1/B8 gate/up/down
cases. Each projection requests exactly 43,520 distinct weight pages with
multiplicity one and no out-of-range reads. Gate/up have 39 active readers in
an 11×4 capacity; down has 32 in 11×3. The per-device layer total is
130,560 software-emitted NoC read requests and **97,484,800 B**
(89,128,960 B payload). B8 occupies one M tile and does not multiply that
weight stream by eight.

The weights remain owned in GDDR allocations across calls. The selected
double-buffered L1 weight CBs contain only 64,512 B/core for gate/up and
87,040 B/core for down; the compute consumes and pops each block. Neither the
tensorbin load cache nor compiled-program reuse is cross-invocation weight
residency. Runtime allocation bases, physical routing/controller commands,
contention, timing and live cache metadata were not measured.

P1.1 therefore closes as a **completed negative characterization**. P1.3 closes
as **rejected**, not as a legal/admitted candidate: bank matching cannot remove
a baseline duplicate that does not exist, and unchanged-dtype placement alone
cannot remove mandatory matrix payload. Candidate request lengths/coalescing,
full implementation legality, and actual performance are not established.
Routing improvements must not be relabeled as byte/reread reduction, and an
unchanged-byte bandwidth interpretation above 512 GB/s remains inadmissible.

P1.2 closes at the explicitly bounded original six-passage B1 diagnostic:
matched equal-six-passage mean delta NLL **≤ +0.010 nats**, equivalent
perplexity ratio **≤ exp(0.010)**. Historical baseline 0.7451044766709086 is
not a universal floor. The [retention screen](evidence/p1-retention-screen-2026-10-25.md)
binds the corpus/provenance, preserves the precision and fused-collective
rejections, and records a real offline score/compare/Boolean-guard smoke with
synthetic passing and failing full-vocabulary exports. No new model retention
measurement is claimed, and the separate historical rounded-register result is
not a bank-matching candidate result.

**P2 must not advance as written.** No prototype, arbitrary successor,
hardware probe, default change, or admission change is introduced. The
requirements and `specs/` remain intact; any separately proposed mechanism
would first have to change aggregate bytes or establish actual reducible
rereads while preserving the numerical and separate matmul/reduce-scatter
contracts. This result does not invent or authorize that work.
