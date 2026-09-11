# Affine group-normalization candidate

This candidate passed the head-row group-normalization weight and bias directly to `ttnn::layer_norm`.

The profiled B4 worker removed `468` no-broadcast multiply launches and `468` no-broadcast add launches. The layer-normalization launch count stayed unchanged. The net reduction was `936` launches.

Physical parity failed before performance measurement:

- Expected short-stream tokens: `[2, 2, 2, 2]`.
- Candidate short-stream tokens: `[59, 2430, 45, 2430]`.
- Expected long-stream prefix: `[39, 6699, 19826, 4706]`.
- Candidate long-stream prefix: `[32053, 32053, 32053, 32053]`.

The fused affine interface does not preserve the existing head-indexed correction semantics. No throughput claim is made. The source change is reverted.

The physical source run remains at `/var/tmp/rwkv7-p150x2-affine-groupnorm-profiler-20260809-2010/` on the measurement host.
