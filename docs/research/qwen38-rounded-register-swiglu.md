# Qwen3.8 rounded register-resident SwiGLU candidate

## Boundary and current status

This is a new, default-off mechanism, not a continuation or admission of the
historically rejected bank-matched weight-residency candidate. It replaces the
**decode/verify gate and up projections plus their product**, not the down
projection, collectives, prefill algorithm, or weight formats. The implementation
is real TT-NN `generic_op` with JIT reader, compute, and writer kernels, installed
in the Qwen package and selected by the actual `Qwen36MLP` consumer.

The installed semantic command passes. The installed Blackhole simulator command
now also passes **exact BF16 equality at B1 and B8**, one iteration each, using
the unchanged installed Qwen/TT-Metal 0.77.0 binaries. TTsim stays at source-pinned
1.10.6 with the bounded `blackhole-pacr-zero-flags-v1` model extension, not a
swap to the native stack's 1.10.9 libraries. There is no simulator-specific
kernel branch. The [dedicated evidence](../../pkgs/qwen36/evidence/2026-10-23/rounded-register-swiglu.json)
preserves both the original unsupported attempt and the completed numerical gate.
This proves the simulated single-chip TP2-local component, not a full model,
multi-chip simulation, silicon performance, or serving admission.

The corrected installed command also passes **exact BF16 product equality** on
physical card 0 for B1 and B8, three repeats each, against the original BF4
fused-SiLU gate plus up/product path. However, synchronized host-wall medians
were **slower**: installed B1 0.411392→0.928172 ms and B8
0.537680→2.236750 ms. These are bounded component/host-dispatch measurements,
not device-profiler throughput. The [physical receipts](../../pkgs/qwen36/evidence/2026-10-23/rounded-register-swiglu-hardware.json)
preserve the local diagnosis and separately repeated installed command. There
is **no promotion** and the default remains off.

No serving admission follows from component compilation or host arithmetic.
`FUSED_MLP_GATEUP_ADMITTED = False` remains unchanged and governs the distinct
prefill AGMM `fuse_swiglu` path. The old MLP-down BF4 rejection remains
**ΔNLL +0.0192 nats > +0.010**, despite its passing top-1 score. See the
[retention-screen inventory](qwen38-retention-screen-evidence.md).

## Exact source and arithmetic contract

The runtime inspected is TT-Metal 0.77.0:

- Source: `/nix/store/5cafvl5cvjzck5hjl36ixj48mpx6j28f-source`.
- Runtime: `/nix/store/7hplkynh047dsk98cwqpfsaaq6lwyy0m-tt-metal-0.77.0`.
- Model source: `d102e7f4512060e09e580e40cf26a870bfa88c55`.
- Intended model-retention checkpoint: `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`.

The [upstream inspiration](https://raw.githubusercontent.com/Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B/95589fb2ebdd7aa400e1daa89dcb3a8d457756bc/scripts/ci/mlp_register_epilogue.py)
implements SiLU in Dst, FP32→BF16 casts of activated gate and up, a BF16 SFPU
product, and final pack. Its adapter is **T16, exactly three pairs per worker,
math-approximate, simulator-only**. That adapter is not transplanted locally.
The local host descriptors, independent gate/up readers, output-column schedule,
mode propagation, and geometry guards are specifically implemented for the
packaged B1/B8 TP2 consumer.

The original packaged 1D decode path uses separate BF4 gate/up weight matrices,
LoFi matmul with FP32 destination accumulation and approximate SFPU math, a
SiLU-fused gate program, and a standalone BF16 product. Exact release sources:

- Model `tt/mlp.py`: the `_mlp_1d_decode` arm chooses the fused-SiLU gate program,
  then the composed path calls `ttnn.mul` on its BF16 output and the BF16 up output.
- Model `tt/model_config.py`: gate/up programs request 44 cores and the physical
  grid width; down requests 33 cores. These configs remain unchanged outside the
  explicitly selected candidate gate/up operation.
- Runtime `operations/matmul/device/kernels/compute/bmm_large_block_zm_fused_bias_activation.cpp`,
  final-output block: activation executes **before** `pack_block`, not after a
  preliminary BF16 rounding of the gate accumulator.
- Runtime `bmm_fused_activation.hpp`: pack-side SiLU operates on Dst; the compute
  API's math-side and pack-side wrappers both invoke `calculate_silu` with
  `DST_ACCUM_MODE` and eight SFPU iterations.
- Blackhole `ckernel_sfpu_typecast.h`: the public FP32→BF16 cast defaults to
  round-to-nearest-even by adding `0x7fff + retained_lsb`, masking the low 16 bits,
  and storing FP32. Its initializer sets Prgm0=1, Prgm1=0x7fff,
  Prgm2=0xffff0000. That is **not** the original native matmul pack tie policy.
- The actual Blackhole `tt-llk/tt_llk_blackhole/common/inc/cpack_common.h`
  configures FP32-Dst reads (`Read_32b_data`) and BF16 pack output. The local
  [pinned ISA packer reference](https://github.com/tenstorrent/tt-isa-documentation/blob/b7738d9ac14a34a4033d60dde9463466b23082e1/WormholeB0/TensixTile/TensixCoprocessor/Packers/FormatConversion.md)
  identifies native nearest/ties-away narrowing; that chapter is WormholeB0,
  so physical Blackhole qualification, rather than assuming cross-architecture
  identity, is decisive.
- The initial public-cast/RNE candidate failed exact physical B1 equality:
  relative L2 0.0001272126974 and maximum absolute error 0.0001220703125.
  Repeated original output was exact. The correction retains the pinned cast
  instruction chain but sets **Prgm0=0 and Prgm1=0x8000**, leaving the mask at
  0xffff0000: `(word + 0x8000) & 0xffff0000`. This gives native-pack ties-away
  at both gate/up boundaries without spilling Dst. Both B1 and B8 then passed
  exact equality on silicon, locally and with the installed command.
- The **product stays RNE**, not ties-away: exact Blackhole
  `ckernel_sfpu_binary.h::calculate_sfpu_binary_mul<..., false>` explicitly uses
  `float32_to_bf16_rne`, matching the original standalone BF16 product.
  The public wrapper would select `DST_ACCUM_MODE=true` for this FP32-Dst
  candidate, so the native BF16 specialization is invoked explicitly.
  The derived call retains Apache-2.0 provenance; surrounding code is local.

The resulting equation is
`RNE_BF16(PACK_BF16(SiLU(gate_FP32)) * PACK_BF16(up_FP32))`, where `PACK_BF16`
uses native nearest/ties-away and the product uses nearest/ties-even, with
original LoFi/FP32 projection arithmetic and approximate device SiLU. It is not
`BF16(SiLU(BF16(gate_FP32)) * BF16(up_FP32))` or an unrounded fused product.
The CPU oracle tests this distinct boundary order and finite tie behavior; it
does not simulate LoFi matmul or SFPU SiLU. The native-pack constant decision is
therefore precision preservation, not a relaxed comparison or a serving claim.

## Selection and explicit rejection

`QWEN38_ROUNDED_REGISTER_SWIGLU=1` selects the candidate for actual decode/verify
MLP consumers. Absent or `0` retains the original implementation. Other values
are errors. Mode is propagated from the actual decoder layer; grouped and
structured-read prefills explicitly identify themselves as `prefill`.
Consequently a process-level opt-in **does not alter prefill**, including short
prefills whose logical height happens to equal B1/B8. No environment toggling is
required between prefill and decode.

Requested decode/verify must satisfy all of:

- Qwen3.8 TP2 model route, original interleaved-weight 1D decode mode;
- Blackhole; one-chip qualification or two-chip SPMD execution at the op layer;
- logical activation `[1,B,5120]` or `[1,1,B,5120]`, **B exactly 1 or 8**;
- tiled BF16 activation and same-device tiled BF4 gate/up weights;
- local weight shape `[5120,8704]` (singleton leading axes allowed);
- original interleaved DRAM weight cache, not DRAM-sharded weights.

Unsupported requested decode rows, widths, formats, layouts, model TP, or mode
raise explicit errors. There is no hidden composed fallback. B16/B32 are not
qualified by padding to the same tile height. The op preserves the logical row
count even though each projection computes one padded tile row.

## What the mechanism changes

Each worker owns disjoint output columns, streams one input tile and the two
corresponding BF4 weight tiles, accumulates gate/up in two FP32 Dst slots,
applies the rounded epilogue, and packs **only the product**. Three output pairs
per worker are not hardcoded. No gate/up intermediate CB or standalone product
launch is present. The down projection and all-reduce stay on the original path.

The existing baseline's gate/up intermediates are **L1**, not DRAM. Eliminating
their write/read removes a conditional `4 × 32 × 8704 × 2 = 2,228,224` bytes of
L1 intermediate traffic per chip/layer for one padded tile row. This is not a
claim of fewer DRAM weight bytes. Under one read per valid weight tile, both
matrices still read `2 × 160 × 272 × 576 = 50,135,040` BF4 bytes per chip/layer.
The simple independent-reader schedule also rereads the activation tile across
output columns instead of using the original multicast schedule. Source-level
intermediate elimination therefore does **not** establish a speedup or a net
bandwidth improvement. Only matched physical profiling can do that.

## Commands and acceptance

New source files must be explicitly staged before the normal Git-worktree flake
can include them. Stage only this candidate's source, never ignored model/runtime
trees (a path flake would copy those private trees into the store):

```console
git add -- pkgs/qwen36/rounded_register_swiglu.py \
  pkgs/qwen36/rounded_register_swiglu_check.py \
  pkgs/qwen36/rounded_register_kernels/compute.cpp \
  pkgs/qwen36/rounded_register_kernels/reader.cpp \
  pkgs/qwen36/rounded_register_kernels/writer.cpp
nix build .#qwen38 --no-link --print-out-paths -L
```

The worker's early scoped builds disabled the broad existing install checks;
those are not normal package acceptance. The final **normal** command
`nix build .#qwen38 --no-link --print-out-paths` passed with all install checks
enabled, producing `/nix/store/sxb2p796i6va54yxpr85xnypkr359xfv-qwen38-0.1.0`.
The new checker wrapper uses Python `-P`: otherwise the build-source `models`
namespace shadows its installed-only module. Package provenance:

- Physically exercised installed component package:
  `/nix/store/9p6q10s8cc4s17k1l2fxpy2d4rjl23p8-qwen38-0.1.0`.
- Consolidated checker package (adds the separate model-smoke mode):
  `/nix/store/87idjn2x4lmsr0nmpjy7dk0q2fai8cvl-qwen38-0.1.0`.
  Parent checksum measurements bind identical numerical module and compute kernel
  bytes across those packages in the physical receipt.

The installed commands are:

```console
$package/bin/qwen38-rounded-register-swiglu-check --semantic
$ttsim/bin/ttsim-blackhole-sfpu \
  $package/bin/qwen38-rounded-register-swiglu-check \
  --device --batch 1 --iterations 1
```

Use the normal `ttsim` package for both simulator batch sizes (`--batch 1` and
`--batch 8`). The accepted final package is
`/nix/store/zyz9cf71yy9gcpnh7cz4r3wk49hvw6yc-ttsim-1.10.6`; its Blackhole library
is SHA-256-identical to the exercised source-built package, as bound in the
receipt. Both baseline and candidate report zero unequal elements and zero
absolute/relative error at K=5120, N=8704.

The original failure was localized **after the candidate returned**, in the
baseline's L1-accumulating packer. The package-matched Blackhole LLK specifies
that `Disable_pack_zero_flags` suppresses outgoing zero metadata, not Dst row
validity. The already supported dense-SRAM simulator path writes actual zeros
and accumulates actual L1 values, so both flag values have the same observable
numerical behavior. The source model now admits that bounded path while keeping
compression, metadata reads, thresholding and Wormhole restrictions intact.
The normal package check exercises the actual PACR implementation over 64
full-face cases, BF16 ties-away narrowing and accumulation transitions. Two
executed defective-model controls fail if validity is bypassed or L1
accumulation is disabled. See [the model boundary](../ttsim-runtime.md#blackhole-pacr-zero-flag-model).
The separate GDN `ALU_ACC_CTRL` limitation is **not** resolved by this extension.

After the parent-owned isolation/reset and firmware compatibility checks,
physical component commands are:

```console
env -u TT_METAL_SIMULATOR -u TT_METAL_SLOW_DISPATCH_MODE \
  $package/bin/qwen38-rounded-register-swiglu-check \
  --device --device-id 0 --batch 1 --iterations 3
env -u TT_METAL_SIMULATOR -u TT_METAL_SLOW_DISPATCH_MODE \
  $package/bin/qwen38-rounded-register-swiglu-check \
  --device --device-id 0 --batch 8 --iterations 3
```

These load **real BF4 weights at the full TP2-local projection geometry**, execute
candidate JIT first, compare against the exact original 44-core fused-SiLU gate
and separate up/product configuration, then repeat both paths to exercise program
reuse. The component criterion is finite **exact BF16 product equality**, not a
guessed relaxed tolerance. Relative L2, maximum absolute error, and unequal value
count are diagnostics on failure. Reported synchronized host-wall medians include
Python/dispatch and are not device-profiler measurements or serving acceptance.

The consolidated package also exposes an actual model continuation smoke using
the existing `BatchedTracedDecodeAdapter`. It keeps the process flag enabled
through unchanged prefills, captures the requested decode width, and carries
recurrent/KV state across the specified steps on physical IDs 0 and 1:

```console
env -u TT_METAL_SIMULATOR -u TT_METAL_SLOW_DISPATCH_MODE \
  QWEN35_GDN_DECODE_FUSION=step QWEN38_ROUNDED_REGISTER_SWIGLU=1 \
  $package/bin/qwen38-rounded-register-swiglu-check \
  --model-device --batch 8 --iterations 2 \
  --model-path "$MODEL" \
  --decode-profile $package/share/qwen36/decode-profile.json
```

This diagnostic emits actual first/continuation token IDs. The corrected source
checker, using installed runtime package `87idjn2x4lmsr0nmpjy7dk0q2fai8cvl`,
completed eight distinct prefills plus two traced recurrent steps on physical
IDs 0 and 1. The process flag stayed enabled and prefill retained its original
route. Exact command, checker SHA-256 and output IDs are in
[`rounded-register-swiglu-model-b8.json`](../../pkgs/qwen36/evidence/2026-10-23/rounded-register-swiglu-model-b8.json).
This is bounded B8 state-carry proof, not B8 distribution/retention admission.

The final installed content-bound tool separately ran fresh original-six B1
baseline and candidate collections on physical IDs 0 and 1, with exact 248320-way
logits and all 385 scored rows. Each run hashed the weight index and all 66
referenced shards before loading and before export. Both six-passage means were
**0.7451044766709086**, with every passage ΔNLL **0.0**; the unchanged original
**+0.010-nat diagnostic passes**. The immutable runtime/package, source-content
manifest, tokenizer/config/corpus identities, complete flags, export/receipt
hashes and commands are in
[`rounded-register-retention.json`](../../pkgs/qwen36/evidence/2026-10-23/rounded-register-retention.json).
Older config-only exports were not silently relabeled: these are new collections
with corrected installed tool `b2wy5v9fkqc2kafybwjyrz5wi2lmx7x1`.

These bounded numerical/model gates pass, but the synchronized installed
component host-wall medians are slower, not a device-profiler speedup. There is
**no performance promotion, serving admission, or default change**. The original
six-passage screen is not a global quality floor, and B8 retention remains
unclaimed. The historical down-BF4 +0.0192 rejection and independent
`FUSED_MLP_GATEUP_ADMITTED=False` remain unchanged.
