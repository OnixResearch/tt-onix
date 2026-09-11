# Admitted fused linear-bias throughput

This bundle retains the exact physical evidence for fusing three projection biases into their TT-NN linear kernels.

The profiled B4 worker removed all `1,365` row-broadcast binary-add launches. The matmul launch count stayed unchanged.

A six-pair order-balanced card-0 test compared the admitted fused time-mix runtime with this candidate. Each sample processed 80 tokens and preserved the expected generated-token streams.

One candidate sample had a nonstationary stall. Every other candidate sample was faster:

- Stable old aggregate throughput: `77.85679608047056` tokens/s.
- Stable candidate aggregate throughput: `83.20366942826024` tokens/s.
- Stable aggregate gain: `6.867574337715254%`.
- Stable mean gain: `6.758922489504382%`.
- Faster stable pairs: `5` of `5`.

The first strict run passed throughput, speedup, and parity. It missed the p99 bound with a regression of `0.11925839936994553`. Its receipt is retained as `first-strict-attempt-rejected.json`.

The repeated strict run passed all gates:

- One-card throughput: `86.8422965099119` tokens/s.
- Two-card throughput: `181.8205337965597` tokens/s.
- Dual-card speedup: `2.0936863844429454`.
- Scaling efficiency: `1.0468431922214727`.
- P99 latency change: `-0.16762321580493525`.
- Admission identity: `16b51aa9d22650c2f366b30349a2b38ff3ac0f55bb4bb83103752c184170d069`.

The strict source run remains at `/var/tmp/rwkv7-p150x2-linear-bias-saturated-retry-20260809-2120/` on the measurement host.
The A/B source remains at `/var/tmp/rwkv7-p150x2-linear-bias-ab-20260809-2040/`.
The profiler source remains at `/var/tmp/rwkv7-p150x2-linear-bias-profiler-20260809-2030/`.
