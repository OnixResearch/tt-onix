# Admitted BF16 WKV output

This bundle retains physical evidence for writing the WKV output matmul directly as BF16.

The recurrent state and matmul inputs remain FP32. Only the matmul output type changes. This removes the separate FP32-to-BF16 cast before attention normalization.

The profiled B4 worker removed `468` typecast launches. Its generated tokens matched the pinned FP32 CPU reference.

A six-pair order-balanced card-0 test included one baseline stall. The five stable pairs measured:

- Baseline aggregate throughput: `79.31003722360683` tokens/s.
- Candidate aggregate throughput: `80.15070832263741` tokens/s.
- Aggregate gain: `1.0599807142447792%`.
- Faster candidate pairs: `4` of `5`.

A four-pair dual-card test included one stalled pair. All three stable candidate rounds were faster:

- Baseline aggregate throughput: `156.365956809559` tokens/s.
- Candidate aggregate throughput: `164.50309268950096` tokens/s.
- Aggregate gain: `5.203905022531408%`.
- Faster candidate pairs: `3` of `3`.

Both strict runs passed admission but contained nonstationary stalls. The retry measured:

- One-card throughput: `68.99105169738961` tokens/s.
- Two-card throughput: `149.08769593833534` tokens/s.
- Dual-card speedup: `2.1609714922489918`.
- P99 latency change: `-0.4801809257635618`.
- Admission identity: `d3a8ca3b483e52a9002376393f034466a1666ebcd0cb300b53d4bf0c39548513`.

The strict absolute result is below the previous `181.8205337965597` tokens/s record. It does not replace that record. The optimization claim uses the alternating A/B evidence.

The retry source remains at `/var/tmp/rwkv7-p150x2-bf16-wkv-output-saturated-retry-20260810-0100/`.
The first strict source remains at `/var/tmp/rwkv7-p150x2-bf16-wkv-output-saturated-20260810-0040/`.
The dual A/B source remains at `/var/tmp/rwkv7-p150x2-bf16-wkv-output-dual-ab-20260810-0120/`.
The card-0 A/B source remains at `/var/tmp/rwkv7-p150x2-bf16-wkv-output-ab-20260810-0020/`.
The profiler source remains at `/var/tmp/rwkv7-p150x2-bf16-wkv-output-profiler-20260810-0010/`.
