# Design

Use device-free semantic checks, simulator-before-hardware, bounded isolated silicon smoke and exact-shape timing. Bind exact package/checkpoint/command and recovered retention interface; preserve NLL +0.010 bound, old BF4 rejection, and FUSED_MLP_GATEUP_ADMITTED=False. No serving admission from component-only evidence.

Evidence distinguishes source contract, numerical execution, physical performance, and unestablished model/serving admission.

## Implemented mechanism and precision decision

The actual 1D decode consumer selects a generic-op pair projection only for
logical B1/B8, K5120, local N8704, existing interleaved BF4 weights on Blackhole
TP2. Explicit prefill mode stays on the original route, even when the process
flag is enabled. Gate/up accumulate in two FP32 Dst slots; SiLU, explicit BF16
rounding of both operands, native BF16 product, and final pack stay in-register.

Exact-release SFPU typecast defaults to nearest-even (`0x7fff + retained_lsb`),
but physical comparison established that preserving original matmul pack means
nearest/ties-away at gate/up boundaries. The local kernel retains the native
cast chain with Prgm0=0, Prgm1=0x8000 and mask0xffff0000; native product remains
RNE. Initial RNE boundary failed exact comparison; corrected local and installed
B1/B8 each passed three repeats exactly. This corrects precision preservation,
not the algorithm, weight dtype, or acceptance threshold.

Gate/up intermediates were L1, not DRAM: elimination does not claim fewer weight
bytes, and the simple reader loses original multicast activation reuse. Physical
host-wall medians were slower at both shapes, so there is no promotion. Details:
`docs/research/qwen38-rounded-register-swiglu.md` and
`pkgs/qwen36/evidence/2026-10-23/rounded-register-swiglu-hardware.json`.

## Simulator resolution, 2026-10-23

The initial `Disable_pack_zero_flags` rejection is historical, not the current
component verdict. Pinned TTsim 1.10.6 now models that flag on its existing
Blackhole dense-L1 path, preserving Dst validity and L1 accumulation. The
unchanged installed Qwen/TT-Metal 0.77.0 kernel passes exact B1/B8 BF16 comparison
on one simulated chip. The [receipt](../../../pkgs/qwen36/evidence/2026-10-23/rounded-register-swiglu.json)
separates `original_simulator_attempt` from `simulator`, including the exercised
library identity and normal package checks. Native TTsim 1.10.9 is unchanged;
the separate GDN `ALU_ACC_CTRL` limitation remains. This is not full-model,
multi-chip, physical-performance or serving evidence; the slower physical
component results still forbid promotion.

