# Fused time-mix admission

This bundle retains the admitted physical result after fused time mixing.

The runtime uses the tensor form of `ttnn::lerp` for each RWKV mix. This operation replaces one subtraction, one multiplication, and one addition.

The model runs seven mixes per layer and recurrent step. The profiled B4 worker replaced `9,828` binary launches with `3,276` ternary launches. This removes `6,552` launches.

The packaged TT-Metal source contains the required ternary compute headers. Its install target omits some headers from the runtime JIT tree. The RWKV closure adds these headers without changing global TT-Metal authority.

A five-pair alternating card-0 test compared the admitted transpose-fusion runtime with the fused runtime. Each sample processed 80 tokens and preserved the expected generated-token streams.

- Old aggregate throughput: `60.774142944169625` tokens/s.
- New aggregate throughput: `82.48592960168224` tokens/s.
- Aggregate change: `35.72536872705587%`.
- Mean throughput change: `35.29735910111405%`.
- Median throughput change: `33.70332391231605%`.
- Faster physical pairs: `5` of `5`.

The strict saturated benchmark passed profiler attribution and exact FP32 CPU-reference token parity.

- One-card throughput: `82.98023822469438` tokens/s.
- Two-card throughput: `171.6697365433619` tokens/s.
- Dual-card speedup: `2.068802647667913`.
- Scaling efficiency: `1.0344013238339564`.
- P99 latency change: `-0.1578332361761085`.
- Admission identity: `de19a9ee3a1cfe9c5d28fdd240485f949c7fb6651fa7189d76b28a611cfa3631`.

Machine latency remains nonstationary. Use the alternating A/B result for the optimization claim. Do not compare separate strict runs as an isolated gain.

The strict source run remains at `/var/tmp/rwkv7-p150x2-lerp-overlay-saturated-20260809-1730/` on the measurement host.
