## Decisions

### Decision: traffic reduction is the candidate, not a demonstrated kernel

**Choice:** investigate MLP gate/up/down weight residency/fusion across the 64
layer calls to reduce per-step weight traffic or rereads. No variant is claimed
to exist or win.

The mechanism decision is pending from the implementation workstream; this
candidate names the investigation, not an implemented variant.

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
