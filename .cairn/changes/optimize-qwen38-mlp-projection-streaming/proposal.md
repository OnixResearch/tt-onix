## Why

MLP weight streaming is the dominant term: 0.3894 ms component median × 64
layers = 24.92 ms of the 48.14 ms component sum, a sum of separately timed
component medians, not an end-to-end step. The traced MLP forward is 335.3 µs
at widths 1 and 16; it is not a projection-only timing ([A], “Component medians”
and “Traced groups”; [B], “Step budget and timing boundaries”).

The reproducible per-device effective weight-byte model is 265.82 GB/s
exponent-exclusive payload / 290.74 GB/s exponent-inclusive tiled storage:
51.9–56.8% of the packaged 512 GB/s P150a ceiling, not measured DRAM traffic.
The verdict's quoted 300–360 GB/s does not reproduce and is corrected in place
by its appended correction ([B], “Weight-byte model and the hardware ceiling”;
[C], “Appended correction”).

## What Changes

- Name the candidate **bank-matched DRAM-sharded weight residency** on the MLP
  gate/up/down path across the 64 layer calls: exactly 8 DRAM reader banks,
  compute grid bound to the 1×8 shard grid, and `per_core_N` derived from it.
  Whether this reduces traffic or rereads remains to be established; not a proven
  winner. Dtype, equations, accumulation, collective order, and every other
  projection's grid remain unchanged.
  Unchanged-bytes 2× implies 531.6–581.5 GB/s, above the rated ceiling ([B],
  “Sensitivity”).
- Require a declared traffic model, device-free evidence, and a numerical-retention
  screen before hardware: fix the NLL/perplexity floor, corpus, and comparison
  method as tasks before implementation, not assumed values.
- Plan a device-free fail-closed legality/rejection module behind a default-off
  admission flag; this records intended work, not an implemented variant.
- Any variant must beat the already-rejected precision option
  `QWEN36_MLP_DOWN_BF4` (NLL rejection) and the
  `matmul_reduce_scatter_decode` decode-M=1 rejection ([A], “Collective share”
  and “What this implies”; [C], “Already-rejected mechanisms”).
- Stage device-free characterization and prototype before a bounded hardware
  probe, deferred until devices 0 and 1 are released; promotion remains subject
  to the existing admission rules.

## Impact

- **Files**: this change's proposal, design, tasks, metadata, and runtime spec only.
- **Testing**: repository-quality artifact and Cairn checks; no hardware or simulator.
- **Risk / non-claims**: no performance, latency, admission, or silicon claim; no
  claim any variant exists. The 24.92 ms is broad MLP attribution, not recoverable
  projection time.

[A]: ../qwen38-tp2-collective-shape-tuning/evidence/step-attribution-2026-10-21.md
[B]: ../../../docs/research/qwen38-mlp-streaming-attribution.md
[C]: ../qwen38-tp2-collective-shape-tuning/evidence/ceiling-verdict-2026-10-22.md
