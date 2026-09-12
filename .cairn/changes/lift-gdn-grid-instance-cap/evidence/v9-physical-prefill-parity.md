# V9: matched physical prefill parity for the chunked scan

Recorded 2026-09-11 for change `lift-gdn-grid-instance-cap`, task V9.

## Result

`/tmp/gdn-grid-hw/v9-scan-parity.py` on physical devices 0 and 1 with the GDN
grid variant runtime. Both dies report 11 x 10 = 110 compute cores.

For each device and for sequence lengths 64 and 256 (two and eight chunks at
the fused chunk size 32), one `ttnn.transformer.chunk_gated_delta_rule` call at
B=8 (192 head-instances, above the core count, so the chunked scan runs) is
compared against eight serialized B=1 calls (24 head-instances, the in-range
mapping) over identical inputs:

```
{"event": "verdict", "all_positions_exact": true}
v9_exit=0
```

Every per-user output and final-state comparison is `torch.equal` (bitwise):
16 comparisons per sequence length per device, 64 exact comparisons total.

Note the serialized per-user reference uses the NV v-blocking search
(`NV = 4` at 24 instances over 110 cores) while the batched call lands on
`NV = 1` with the instance chunking. The two decompositions are bitwise
identical per position, which is what the exact comparison establishes.

## Diagnostic timings (no claim)

| Device | T | batched call | eight serialized calls |
| --- | ---: | ---: | ---: |
| 0 | 64 | 4.454 s (first call, includes JIT) | 0.996 s |
| 0 | 256 | 0.648 s | 0.323 s |
| 1 | 64 | 0.018 s | 0.015 s |
| 1 | 256 | 0.018 s | 0.022 s |

Device 0's first sample includes kernel compilation; the steady-state pairs sit
in the same tens-of-milliseconds band, with the batched call at or below the
serialized total on device 1. These are op-level diagnostics only. The profile
makes no timing claim, and the served path still prefills per user, so this
evidence does not predict a served TTFT change by itself.

## What this covers

The service never launches a scan with `BH` above the core count today: its
prefills are per user (`prefill_paged_slots`, the traced-bucket path replays a
B=1 trace per user). The guard on the scan factory is nevertheless what forces
that design, so V9 exercises the lifted guard directly at the op level rather
than waiting for a service path that does not exist yet.

## Non-claims

- No admission and no timing claim.
- No claim about the served batched-prefill path; only the op-level mechanism
  is measured here.
