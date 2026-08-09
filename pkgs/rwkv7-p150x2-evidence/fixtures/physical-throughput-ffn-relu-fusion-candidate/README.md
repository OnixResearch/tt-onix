# FFN ReLU fusion candidate

This candidate fused FFN ReLU into its preceding matmul. An explicit full-device core grid was required for TT-NN to fuse the activation.

The profiled B4 worker removed `468` unary launches and preserved the pinned generated tokens. Its elapsed time increased from `0.981` to `1.125986876` seconds.

A four-pair order-balanced physical test measured:

- Baseline aggregate throughput: `78.48606671881947` tokens/s.
- Candidate aggregate throughput: `76.68800326945498` tokens/s.
- Aggregate change: `-2.2909333140697985%`.
- Faster candidate pairs: `1` of `4`.

The explicit grid changes matmul scheduling enough to cost more than the removed unary dispatch. The source change is reverted.

The A/B source remains at `/var/tmp/rwkv7-p150x2-ffn-relu-ab-20260809-2340/`.
The profiler source remains at `/var/tmp/rwkv7-p150x2-ffn-relu-profiler-20260809-2320/`.
