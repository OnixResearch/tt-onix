# Head-row normalization admission

This bundle retains the admitted physical result after head-row normalization.

The runtime loads static attention correction vectors in head-row shape. It applies group-normalization scale and bias directly to normalized head rows.

This change removes three reshape launches per layer and recurrent step. The profiled B4 worker reduced tiled reshape launches from `6,591` to `5,187`. This is a reduction of `1,404`, or `21.301775147928993%`.

A five-pair alternating card-0 test compared the BF16 row-reuse runtime with the new runtime. Each sample processed 80 tokens and preserved the expected generated-token streams.

- Old aggregate throughput: `56.091624936777606` tokens/s.
- New aggregate throughput: `61.17529133445523` tokens/s.
- Aggregate change: `9.063146955374467%`.
- Mean throughput change: `8.030536019364964%`.
- Median throughput change: `15.493271183032853%`.

The strict saturated benchmark also passed profiler attribution and exact FP32 CPU-reference token parity.

- One-card throughput: `62.69137518901232` tokens/s.
- Two-card throughput: `125.85112395302231` tokens/s.
- Dual-card speedup: `2.007471100667126`.
- Scaling efficiency: `1.003735550333563`.
- P99 latency change: `-0.11914938276738306`.
- Admission identity: `9f89f074fe1cf1dc93484d8eb51f017c2692eef092ccf4e75ab78887f4d04034`.

Machine latency remains nonstationary. Use the alternating A/B result for the optimization claim. Do not treat separate strict-run differences as isolated gains.

The strict source run remains at `/var/tmp/rwkv7-p150x2-head-row-normalization-saturated-20260809-1030/` on the measurement host.
