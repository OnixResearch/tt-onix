# Qwen3.8 Rounded-Register SwiGLU Precision Plan

This document specifies the arithmetic contract, pinned instruction mapping, domain
boundaries, unknown hardware behaviors, and full-kernel feasibility inventory for the
opt-in Blackhole TP2 decode gate/up SwiGLU candidate
([`rounded_register_swiglu.py`](../pkgs/qwen36/rounded_register_swiglu.py),
[`compute.cpp`](../pkgs/qwen36/rounded_register_kernels/compute.cpp)).

It accompanies—and does not weaken or replace—the existing exact-parity (`torch.equal`)
device qualification gate in
[`rounded_register_swiglu_check.py`](../pkgs/qwen36/rounded_register_swiglu_check.py).
No production numerics or admission gates are changed by this plan.

## Pinned Source and Runtime Bindings

- Candidate host wrapper: `pkgs/qwen36/rounded_register_swiglu.py`
- Candidate compute kernel: `pkgs/qwen36/rounded_register_kernels/compute.cpp`
- Qualification checker: `pkgs/qwen36/rounded_register_swiglu_check.py`
- Independent epilogue oracle: `pkgs/qwen36/rounded_register_swiglu_oracle.py`
- Pinned TT-Metal runtime: `v0.78.0` (`pkgs/tt-metal/default.nix`, source hash
  `sha256-Gd8ZdAxS54JBNasPI8KfXCrXXw8CwweYaW/Y0ivBn/E=`)
- Pinned SFPI toolchain: `7.71.0` (`pkgs/tt-metal/default.nix`)
- Architecture and geometry: Blackhole, single qualification chip or TP2 (`1x2`),
  logical decode batch `B in {1, 8}`, `K = 5120` (160 tiles), local `N = 8704` (272 tiles).

## Stage-by-Stage Operation and Instruction Map

The compute kernel (`pkgs/qwen36/rounded_register_kernels/compute.cpp`) runs with
`ComputeConfigDescriptor(math_fidelity=LoFi, fp32_dest_acc_en=True, math_approx_mode=True, dst_full_sync_en=False)`.

| Stage | Local call site | Pinned TT-Metal `v0.78.0` path | Input / Output format | Rounding / tie rule & flags | Domain / overflow / subnormal status |
| --- | --- | --- | --- | --- | --- |
| 0. Weight quantization (offline/cache) | `ttnn.from_torch(..., dtype=ttnn.bfloat4_b)` (pre-kernel cache) | TT-NN `Bfp4_b` block packer | BF16 -> `Bfp4_b` (shared 8-bit exp per 16-elem face row, 3-bit signed mag mantissa) | Pre-kernel source quantization; **distinct** from kernel arithmetic error after stored tiles are chosen | Out of scope for epilogue oracle |
| 1. Gate & Up matmuls (`K=160` tiles) | `compute.cpp:18-27` (`matmul_init(0,1)`, `matmul_tiles(0,1,0,0,0)` & `matmul_tiles(0,1,0,1,1)`) | `tt_metal/hw/inc/api/compute/matmul.h` -> Blackhole LLK FPU matmul | CB0 `Float16_b`, CB1 `Bfp4_b` -> Dst0/Dst1 `Float32` (`fp32_dest_acc_en=True`) | `MathFidelity::LoFi` (1-phase truncated mantissa multiply), accumulated sequentially over `k = 0..159` into FP32 Dst | **Unknown / hardware-owned**: exact FPU LoFi mantissa truncation, internal accumulation rounding, subnormal/FTZ, and overflow saturation |
| 2. Gate SiLU activation | `compute.cpp:30-31` (`silu_tile_init()`, `silu_tile(0)`) | `compute_kernel_api.h:163-175` -> `llk_sfpu/ckernel_sfpu_silu.h:14-36` (`calculate_silu<true, 8>`) -> `ckernel_sfpu_sigmoid.h:18-40` | Dst0 `Float32` -> Dst0 `Float32` | Computes `x * _sfpu_sigmoid_<true>(x)` using `_sfpu_exp_accurate_<true>(-x)` and 2-step Newton-Raphson `sfpu_reciprocal_iter<2>(1 + exp(-x))`; `silu_init` forces `sigmoid_init<false>()` regardless of `math_approx_mode=True`. No BF16 cast because `is_fp32_dest_acc_en=true` | **Device-owned / unmodeled**: SFPU polynomial/exp/recip ULP error, subnormal FTZ, extreme negative/positive saturation |
| 3a. Intermediate gate cast | `compute.cpp:34-41` (`typecast_tile_init<fp32, bf16>()`, `vConstIntPrgm0 = 0`, `vConstIntPrgm1 = 0x8000`, `typecast_tile<fp32, bf16>(0)`) | `llk_sfpu/ckernel_sfpu_typecast.h:247-262, 580-586` (`calculate_typecast_fp32_to_fp16b`) | Dst0 `Float32` -> Dst0 BF16-in-`Float32` (low 16 bits zeroed) | Overrides default RNE constants (`1`, `0x7fff`) to (`0`, `0x8000`), keeping `LREG14 = 0xffff0000`. Executes `((word + 0x8000) & 0xffff0000)` in 32-bit integer arithmetic: **Round-to-Nearest, Ties-Away-from-Zero (RNA)** on finite normal FP32 values | Modeled by epilogue oracle for normal finite inputs & `±0.0`; nonfinite, overflow (`|x| >= 2^128 - 2^119`), and FP32/BF16 subnormals are classified `out_of_domain` / `inconclusive` |
| 3b. Intermediate up cast | `compute.cpp:42` (`typecast_tile<fp32, bf16>(1)`) | Same as 3a | Dst1 `Float32` -> Dst1 BF16-in-`Float32` | Identical **ties-away (RNA)** `((word + 0x8000) & 0xffff0000)` rule as 3a | Same as 3a |
| 4. BF16 product in FP32 Dst | `compute.cpp:43-48` (`mul_binary_tile_init()`, `SFPU_BINARY_CALL(..., calculate_sfpu_binary_mul, (APPROX, MUL, 8, false), 0, 1, 0, RC)`) | `llk_sfpu/ckernel_sfpu_binary.h:148-169` + `llk_sfpu/ckernel_sfpu_conversions.h:34-55` (`float32_to_bf16_rne`) | Dst0, Dst1 BF16-in-`Float32` -> Dst0 BF16-in-`Float32` | 1. Exact FP32 product `in0 * in1` (exact for normal finite BF16 operands since 8×8 = 16 ≤ 24 significand bits). 2. `is_fp32_dest_acc_en=false` triggers `float32_to_bf16_rne`: `lsb = (bits >> 16) & 1; (bits + 0x7fffU + lsb) & 0xffff0000U` — **IEEE-754 Round-to-Nearest, Ties-to-Even (RNE)** (distinct from stage 3!). 3. Zero-product canonicalization: `v_if(in0 == 0 \|\| in1 == 0) { result = 0.0f; }` forces `+0.0` | Modeled by epilogue oracle for normal finite products & zero products; product overflow (`|g*u| >= 2^128 - 2^119`) and subnormal underflow (`0 < |g*u| < 2^-126`) are classified `out_of_domain` / `inconclusive` |
| 5. Final tile store | `compute.cpp:49-54` (`pack_tile(0, 16)`) | `tt_metal/hw/inc/api/compute/tile_move_copy.h` -> Blackhole packer (`Float32` Dst -> `Float16_b` CB16) | Dst0 BF16-in-`Float32` -> CB16 `Float16_b` | Lossless upper-16-bit pack on already-rounded BF16-in-FP32 values (low 16 bits are `0x0000`, so packer rounding cannot carry on normal finite values) | Modeled as lossless identity on in-domain BF16-in-FP32 values; hardware subnormal FTZ during pack is **unknown** |

## Critical Arithmetic Findings from the Pinned Source

1. **Asymmetric tie rules across stages**:
   - Stages 3a and 3b (`typecast_tile` with overridden `vConstIntPrgm0 = 0`, `vConstIntPrgm1 = 0x8000`)
     use **ties-away-from-zero (RNA)** to match the baseline matmul's native packer bias.
   - Stage 4 (`calculate_sfpu_binary_mul<..., false>`) calls `float32_to_bf16_rne`, which uses
     **ties-to-even (RNE)**.
   - Assuming one uniform tie rule across the entire epilogue is mathematically wrong and is
     distinguished by exact witness pairs such as `(gate_act = 1.5, up = 1 + 3 * 2^-7)`.
2. **Signed-zero canonicalization in Stage 4**:
   - `ckernel_sfpu_binary.h:163` explicitly forces `result = 0.0f` (`0x0000`) whenever either
     rounded operand equals zero, matching Tenstorrent FPU BF16 multiplication (`(-0.0) * (+3.0) = +0.0`).
   - PyTorch's `(activated.float() * rounded_up.float()).to(torch.bfloat16)` in
     `rounded_register_swiglu.py:60` produces IEEE `-0.0` (`0x8000`) on `(-0.0) * (+3.0)` or
     `(+0.0) * (-3.0)`. `torch.equal` reports value equality (`-0.0 == +0.0`), while bit-exact
     checking makes this distinction explicit.
3. **Wrap-around hazard on nonfinite inputs in Stage 3**:
   - The raw 32-bit integer addition `word + 0x8000` in `calculate_typecast_fp32_to_fp16b`
     does not guard against `NaN`/`Inf` (e.g., `0x7FFFFFFF + 0x8000` wraps to `0x80000000`).
   - Therefore nonfinite inputs and overflow candidates must be classified `out_of_domain`,
     never silently accepted.

## Unknown / Undocumented Hardware Details

The following hardware behaviors are **not** proven from host source alone and remain
explicitly marked **unknown**:

- Blackhole Tensix FPU exact mantissa truncation/guard bits under `MathFidelity::LoFi`
  for `Float16_b x Bfp4_b -> Float32` tile products.
- Exact FPU accumulation tree order and intermediate rounding across the `K = 160` tile loop.
- Blackhole SFPU and packer flush-to-zero (FTZ) vs preserved-subnormal behavior on
  `SFPLOAD`, `SFPSTORE`, `SFPMUL`, and `pack_tile`.
- Exact worst-case ULP error of `_sfpu_exp_accurate_<true>` and `sfpu_reciprocal_iter<2>`
  across all finite FP32 inputs on Blackhole silicon.

## Full-Kernel Feasibility Inventory (Blocked Operations)

This plan and `rounded_register_swiglu_oracle.py` establish an **independent exact-bit/rational
epilogue oracle** for Stages 3a, 3b, 4, and 5 only. Extending to a full-kernel mathematical
error enclosure requires resolving three currently blocked components:

1. **BF4 block quantization (Stage 0)**:
   - *Distinction*: Errors introduced when converting original weights to `Bfp4_b` belong to
     weight representation / model quantization, not kernel execution error relative to the
     stored `Bfp4_b` weights.
   - *Needed*: Exact host bit-unpacking of `Bfp4_b` tiles (matching TT-Metal's 16-element shared
     exponent layout) so kernel references consume the exact stored `Bfp4_b` values as rational inputs.
2. **LoFi matmul accumulation (Stage 1)**:
   - *Needed*: Reviewed Blackhole ISA / FPU specification for `LoFi` mantissa truncation bit
     positions, partial-product rounding, and intra-tile / cross-K accumulation order, plus a
     dedicated hardware probe exposing `matmul_tiles` outputs on known exact `Bfp4_b` inputs.
3. **SFPU SiLU approximation (Stage 2)**:
   - *Needed*: Exact polynomial/range-reduction interval error proof for
     `_sfpu_exp_accurate_<true>` and 2-step Newton-Raphson `sfpu_reciprocal_iter<2>` under
     Blackhole SFPU `SFPMAD` rounding (and hardware confirmation that silicon matches the
     pinned C++ header). Empirical silicon calibration can bound observed inputs only; it is
     **not** a universal error theorem and must not be used to widen tolerances until an
     existing kernel passes.

Until Stages 1 and 2 have verified hardware/ISA contracts, full-kernel qualification keeps
the existing exact-parity (`torch.equal`) device gate unchanged while this oracle provides
independent mathematical verification of the epilogue (Stages 3–5).

## Reporting Separation and Cost Accounting

Every qualification report must keep these sections strictly separate:

1. **Exact parity**: baseline vs candidate `torch.equal` and finite checks (unchanged).
2. **Independent epilogue arithmetic oracle**: exact-bit/rational verification of Stages 3–5
   with explicit `scope = "epilogue_only_not_silu_or_matmul"`.
3. **Domain coverage**: explicit counts of `in_domain`, `out_of_domain_nonfinite`,
   `out_of_domain_overflow`, and `inconclusive_subnormal_ftz_unmodeled`.
4. **Mutation & common-mode controls**: witness-backed detection of every injected epilogue
   fault and proof that pairwise equality alone fails on common-mode bugs.
5. **Unresolved assumptions**: explicit enumeration of unknown hardware behaviors and blocked
   full-kernel stages.
6. **Checking cost**: CPU oracle / verification wall time reported separately from kernel
   device wall time and never subtracted from or conflated with kernel timing.
