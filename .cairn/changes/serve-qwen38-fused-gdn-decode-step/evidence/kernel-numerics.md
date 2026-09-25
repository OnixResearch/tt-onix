# Fused step programs against a float64 reference

Recorded 2026-09-24 on the P150x2 mesh, with the same programs replicated on
both devices. This run is physical. It checks one layer and makes no model-level
claim.

The throwaway test is `.pi/tmp/gdn-step/test_step.py`. It uses the Qwen3.8 TP=2
layer geometry: 8 key heads, 24 value heads, 128-wide heads, and a 5120-wide
q/k/v block. Inputs are random, stored as bf16 or fp32 exactly as the model
stores them. The reference is `decode_step.reference_step`, computed in
float64 from the same device-rounded inputs.

| Case | Conv state shift | Conv output (fp32) | Recurrent state rel. error | Gated output rel. error | Idle rows untouched |
| --- | --- | ---: | ---: | ---: | --- |
| B=1, capacity 1 | bit-exact | 6.1e-08 | 1.24e-03 | 1.89e-03 | n/a |
| B=4, capacity 4 | bit-exact | – | 1.94e-03 | 1.97e-03 | n/a |
| B=3, capacity 8 | bit-exact | – | 1.35e-03 | 1.90e-03 | yes |

The gated output is bf16, so its error floor is bf16 rounding (about 2e-3). The
recurrent-state error comes from the tf32 operands of the rank-1 update and the
`k S'` read. The carried term `decay * S` is computed in fp32 DEST with fp32 SFPU
multiplication.

## Rejected precision variant and a hardware finding

The first version routed the state through the FPU (`mul_tiles_bcast_scalar`
and `copy_tile` from default-mode CBs), as the existing `gated_delta_decode`
kernel does. That path truncates the carried state to tf32 on every step. It
measured a state error of 2.1e-03 on the same case. That error is systematic
shrinkage, which compounds like an extra decay factor. The exact-DEST path above
removes it.

While building that path, a microkernel on silicon showed a failure in scalar
broadcast. Unpack-to-DEST scalar broadcast (`unary_bcast<SCALAR>` from a CB in
`UnpackToDestFp32` mode) does not broadcast after SFPU state has been programmed
in the kernel. It copies the whole tile instead, and it corrupts a neighbouring
DEST tile. The same broadcast works when it is the first operation. Scalars
therefore take the source-register broadcast path from default-mode CBs. Every
broadcast scalar is either bf16-exact (a, b, `dt_bias`, `-exp(A)`) or scales only
a bf16 output (`1/rms(o)`).

## Timing (traced, back-to-back replays, B=1)

| Program | µs |
| --- | ---: |
| `conv_prep` (258 tiles over 86 cores) | 23.9 |
| `recurrence` (24 heads, one per core) | about 48 |
| both | 72.4 |

The composite glue that these programs replace costs 233 µs per layer in the
fused route and 308 µs in the composite route (`attribution.md`). Two
dispatch facts mattered. Per-core single ranges cost about twice as much as
rectangular ranges (`conv_prep` took 55 µs as 86 unicast cores). Double-buffered
CBs also helped.
