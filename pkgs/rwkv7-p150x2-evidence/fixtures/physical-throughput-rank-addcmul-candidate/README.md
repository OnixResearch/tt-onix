# Rank-update addcmul candidate

This candidate replaced the FP32 rank-update matmul and its following state add with one mixed-broadcast `ttnn::addcmul` operation.

The profiled B4 worker removed `468` matmul launches and `468` binary-add launches. It added `468` row-and-column broadcast ternary launches. The net reduction was `468` launches.

Two counterbalanced five-pair card-0 tests compared the admitted fused time-mix runtime with this candidate. Each sample processed 80 tokens and preserved the expected generated-token streams.

- Old aggregate throughput: `77.98535226181258` tokens/s.
- Candidate aggregate throughput: `77.75328945809615` tokens/s.
- Aggregate change: `-0.2975722965735228%`.
- Mean throughput change: `-0.3699266845228033%`.
- Faster candidate pairs: `5` of `10`.

The launch reduction did not produce a physical gain. The source change is reverted.

The forward A/B source remains at `/var/tmp/rwkv7-p150x2-rank-addcmul-ab-20260809-1920/`.
The reverse-order source remains at `/var/tmp/rwkv7-p150x2-rank-addcmul-reverse-ab-20260809-1935/`.
