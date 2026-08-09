# Mixed `a` and `b` candidate

This candidate passed BF16 `a` and `b` vectors directly to FP32 recurrent-state matmuls.

It removed `936` typecast launches from one profiled B4 worker. Generated tokens matched the pinned FP32 CPU reference.

The profiler elapsed time increased from `0.932745777` to `1.150330766` seconds.

All four order-balanced physical pairs regressed:

- Baseline aggregate throughput: `81.37537238676862` tokens/s.
- Candidate aggregate throughput: `77.19304035769194` tokens/s.
- Aggregate change: `-5.139555010818864%`.
- Faster candidate pairs: `0` of `4`.

Mixed inputs select a slower matmul path for these state-update products. The source change is reverted.

The A/B source remains at `/var/tmp/rwkv7-p150x2-mixed-ab-ab-20260810-0350/`.
The profiler source remains at `/var/tmp/rwkv7-p150x2-mixed-ab-profiler-20260810-0340/`.
