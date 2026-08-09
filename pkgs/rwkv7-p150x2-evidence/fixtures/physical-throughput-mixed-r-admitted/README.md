# Admitted mixed-precision read vector

This bundle retains physical evidence for passing the BF16 read vector directly to the WKV output matmul.

The recurrent state remains FP32. The matmul output remains explicitly BF16. The change removes the separate BF16-to-FP32 read-vector cast.

The profiled B4 worker removed `468` typecast launches. Its generated tokens matched the pinned FP32 CPU reference.

A six-pair order-balanced card-0 test included one baseline stall. Every stable candidate sample was faster:

- Baseline aggregate throughput: `80.13932185843787` tokens/s.
- Candidate aggregate throughput: `83.86687211405733` tokens/s.
- Aggregate gain: `4.651337407376599%`.
- Faster candidate pairs: `5` of `5`.

Strict admission passed:

- One-card throughput: `73.54229630441628` tokens/s.
- Two-card throughput: `152.99240618044047` tokens/s.
- Dual-card speedup: `2.0803321879854484`.
- Scaling efficiency: `1.0401660939927242`.
- P99 latency change: `-0.453708148495063`.
- Admission identity: `8c65d0fe7f1b39907a100fac0457bb4ab128b9c3b1fcdd58aa71466ee29aa93c`.

The strict run contains nonstationary stalls and does not replace the `181.8205337965597` tokens/s absolute record. The optimization claim uses the alternating A/B evidence.

Mixed input types can select a different matmul compute configuration. Exact token parity is proven only for the pinned benchmark workload.

The strict source remains at `/var/tmp/rwkv7-p150x2-mixed-r-saturated-20260810-0240/`.
The A/B source remains at `/var/tmp/rwkv7-p150x2-mixed-r-ab-20260810-0220/`.
The profiler source remains at `/var/tmp/rwkv7-p150x2-mixed-r-profiler-20260810-0210/`.
