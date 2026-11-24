"""Independent exact-bit/rational CPU epilogue oracle for rounded-register SwiGLU.

SCOPE: Epilogue ONLY (Stages 3a, 3b, 4, and 5 of docs/qwen38-rounded-register-swiglu-precision-plan.md):
  - Stage 3a/3b: FP32 -> BF16 ties-away-from-zero (RNA) intermediate casts
  - Stage 4:     Exact rational BF16 x BF16 product -> BF16 ties-to-even (RNE)
                 with Tenstorrent FPU/SFPU zero-product canonicalization to +0.0
  - Stage 5:     Lossless pack_tile of BF16-in-FP32 Dst register to BF16 CB16

This module is NOT a SiLU oracle and NOT a matmul or BF4-quantization oracle.
It intentionally does NOT import rounded_register_swiglu.py or native_pack.
"""

from __future__ import annotations

import hashlib
import struct
import time
from collections.abc import Callable
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

ORACLE_SCOPE = "epilogue_only_not_silu_or_matmul"
PRECISION_PLAN_RELATIVE_PATH = "docs/qwen38-rounded-register-swiglu-precision-plan.md"
PINNED_TT_METAL_VERSION = "0.78.0"

DOMAIN_IN_DOMAIN = "in_domain"
DOMAIN_NONFINITE = "out_of_domain_nonfinite"
DOMAIN_OVERFLOW = "out_of_domain_overflow"
DOMAIN_SUBNORMAL_UNMODELED = "inconclusive_subnormal_ftz_unmodeled"


@dataclass(frozen=True)
class EpilogueElementResult:
    """Exact rational/bit-level epilogue specification for one (gate_act, up) pair."""

    gate_act_f32_bits: int
    up_f32_bits: int
    domain_status: str
    gate_bf16_bits: int | None
    up_bf16_bits: int | None
    expected_bf16_bits: int | None
    ieee_signed_zero_bf16_bits: int | None
    exact_product_rational: Fraction | None
    scope: str = ORACLE_SCOPE


def f32_to_bits(value: float) -> int:
    """Convert Python float to IEEE-754 binary32 bit pattern (unsigned 32-bit int)."""
    return struct.unpack(">I", struct.pack(">f", float(value)))[0]


def bits_to_f32(bits: int) -> float:
    """Convert unsigned 32-bit IEEE-754 binary32 pattern to Python float."""
    return struct.unpack(">f", struct.pack(">I", bits & 0xFFFFFFFF))[0]


def bf16_bits_to_f32(bf16_bits: int) -> float:
    """Convert unsigned 16-bit BF16 pattern to Python float via upper-16 FP32 bits."""
    return bits_to_f32((bf16_bits & 0xFFFF) << 16)


def f32_bits_to_fraction(bits: int) -> tuple[Fraction | None, int, str]:
    """Decode IEEE-754 binary32 bits to (exact Fraction, sign_bit, domain_status)."""
    u32 = bits & 0xFFFFFFFF
    sign = (u32 >> 31) & 1
    exp = (u32 >> 23) & 0xFF
    mant = u32 & 0x7FFFFF
    if exp == 0xFF:
        return None, sign, DOMAIN_NONFINITE
    if exp == 0:
        if mant == 0:
            return Fraction(0), sign, DOMAIN_IN_DOMAIN
        return None, sign, DOMAIN_SUBNORMAL_UNMODELED
    unbiased = exp - 127
    sig = (1 << 23) | mant
    shift = unbiased - 23
    mag = Fraction(sig << shift, 1) if shift >= 0 else Fraction(sig, 1 << (-shift))
    return (-mag if sign else mag), sign, DOMAIN_IN_DOMAIN


def bf16_bits_to_fraction(bf16_bits: int) -> tuple[Fraction | None, int, str]:
    """Decode unsigned 16-bit BF16 bits to (exact Fraction, sign_bit, domain_status)."""
    return f32_bits_to_fraction((bf16_bits & 0xFFFF) << 16)


def _floor_log2_positive_fraction(mag: Fraction) -> int:
    """Return exact integer floor(log2(mag)) for mag > 0."""
    num = mag.numerator
    den = mag.denominator
    exp = num.bit_length() - den.bit_length()
    if exp >= 0:
        if num < (den << exp):
            exp -= 1
    elif (num << (-exp)) < den:
        exp -= 1
    return exp


def round_fraction_to_bf16_bits(
    value: Fraction, sign_bit: int, tie_mode: str
) -> tuple[int | None, str]:
    """Round an exact rational value to normal BF16 bits under an explicit tie rule.

    Supported tie modes:
      - "ties_away": Round-to-Nearest, Ties-Away-from-Zero (RNA)
      - "ties_even": Round-to-Nearest, Ties-to-Even (RNE)
      - "truncate":  Round-toward-zero (truncation)
    """
    if tie_mode not in ("ties_away", "ties_even", "truncate"):
        raise ValueError(f"unsupported tie_mode: {tie_mode!r}")
    if value == 0:
        return (sign_bit & 1) << 15, DOMAIN_IN_DOMAIN
    mag = abs(value)
    exp = _floor_log2_positive_fraction(mag)
    if exp < -126:
        return None, DOMAIN_SUBNORMAL_UNMODELED
    if exp > 127:
        return None, DOMAIN_OVERFLOW

    scale_shift = 7 - exp
    scaled = (
        mag * (1 << scale_shift)
        if scale_shift >= 0
        else mag / Fraction(1 << (-scale_shift))
    )
    q_int = scaled.numerator // scaled.denominator
    rem = scaled - q_int
    half = Fraction(1, 2)

    if tie_mode == "ties_away":
        round_up = rem >= half
    elif tie_mode == "ties_even":
        round_up = rem > half or (rem == half and (q_int & 1) == 1)
    else:
        round_up = False

    mant_full = q_int + (1 if round_up else 0)
    if mant_full == 256:
        mant_full = 128
        exp += 1
    if exp > 127:
        return None, DOMAIN_OVERFLOW

    bf16_bits = ((sign_bit & 1) << 15) | ((exp + 127) << 7) | (mant_full & 0x7F)
    return bf16_bits, DOMAIN_IN_DOMAIN


def epilogue_oracle_element(
    gate_act_f32_bits: int, up_f32_bits: int
) -> EpilogueElementResult:
    """Compute exact specification of Stages 3a, 3b, 4, and 5 on one FP32 pair."""
    g_frac, g_sign, g_dom = f32_bits_to_fraction(gate_act_f32_bits)
    if g_dom != DOMAIN_IN_DOMAIN or g_frac is None:
        return EpilogueElementResult(
            gate_act_f32_bits, up_f32_bits, g_dom, None, None, None, None, None
        )
    u_frac, u_sign, u_dom = f32_bits_to_fraction(up_f32_bits)
    if u_dom != DOMAIN_IN_DOMAIN or u_frac is None:
        return EpilogueElementResult(
            gate_act_f32_bits, up_f32_bits, u_dom, None, None, None, None, None
        )

    # Stages 3a & 3b: FP32 -> BF16 ties-away-from-zero (RNA).
    g_bf16, g_cast_dom = round_fraction_to_bf16_bits(g_frac, g_sign, "ties_away")
    if g_cast_dom != DOMAIN_IN_DOMAIN or g_bf16 is None:
        return EpilogueElementResult(
            gate_act_f32_bits, up_f32_bits, g_cast_dom, None, None, None, None, None
        )
    u_bf16, u_cast_dom = round_fraction_to_bf16_bits(u_frac, u_sign, "ties_away")
    if u_cast_dom != DOMAIN_IN_DOMAIN or u_bf16 is None:
        return EpilogueElementResult(
            gate_act_f32_bits, up_f32_bits, u_cast_dom, g_bf16, None, None, None, None
        )

    g_bf16_frac, _, _ = bf16_bits_to_fraction(g_bf16)
    u_bf16_frac, _, _ = bf16_bits_to_fraction(u_bf16)
    assert g_bf16_frac is not None and u_bf16_frac is not None

    # Stage 4: Exact BF16 x BF16 rational product, RNE round, zero canonicalization.
    exact_prod = g_bf16_frac * u_bf16_frac
    ieee_sign = g_sign ^ u_sign
    if exact_prod == 0:
        # ckernel_sfpu_binary.h:163 forces +0.0f when either operand is zero.
        return EpilogueElementResult(
            gate_act_f32_bits=gate_act_f32_bits,
            up_f32_bits=up_f32_bits,
            domain_status=DOMAIN_IN_DOMAIN,
            gate_bf16_bits=g_bf16,
            up_bf16_bits=u_bf16,
            expected_bf16_bits=0x0000,
            ieee_signed_zero_bf16_bits=ieee_sign << 15,
            exact_product_rational=exact_prod,
        )

    prod_bf16, prod_dom = round_fraction_to_bf16_bits(
        exact_prod, ieee_sign, "ties_even"
    )
    if prod_dom != DOMAIN_IN_DOMAIN or prod_bf16 is None:
        return EpilogueElementResult(
            gate_act_f32_bits=gate_act_f32_bits,
            up_f32_bits=up_f32_bits,
            domain_status=prod_dom,
            gate_bf16_bits=g_bf16,
            up_bf16_bits=u_bf16,
            expected_bf16_bits=None,
            ieee_signed_zero_bf16_bits=None,
            exact_product_rational=exact_prod,
        )

    # Stage 5: pack_tile(0, 16) is lossless on BF16-in-FP32.
    return EpilogueElementResult(
        gate_act_f32_bits=gate_act_f32_bits,
        up_f32_bits=up_f32_bits,
        domain_status=DOMAIN_IN_DOMAIN,
        gate_bf16_bits=g_bf16,
        up_bf16_bits=u_bf16,
        expected_bf16_bits=prod_bf16,
        ieee_signed_zero_bf16_bits=prod_bf16,
        exact_product_rational=exact_prod,
    )


def build_discriminating_fixtures() -> dict[str, list[tuple[int, int]]]:
    """Deterministic discriminating (gate_act_f32_bits, up_f32_bits) families."""

    def pair(g: float, u: float) -> tuple[int, int]:
        return (f32_to_bits(g), f32_to_bits(u))

    return {
        # Positive half-way ties (even-LSB and odd-LSB) for both gate_act and up.
        "positive_halfway_ties": [
            pair(1.00390625, 1.0),  # 1 + 2^-8 (even LSB 0 -> RNA 1.0078125, RNE 1.0)
            pair(1.01171875, 1.0),  # 1 + 3*2^-8 (odd LSB 1 -> both 1.015625)
            pair(1.0, 1.00390625),
            pair(1.0, 1.01171875),
            pair(128.5, 1.00390625),
            pair(129.5, 1.01171875),
            pair(1.00390625, 1.00390625),
        ],
        # Negative half-way ties (RNA rounds away from zero: -1.00390625 -> -1.0078125).
        "negative_halfway_ties": [
            pair(-1.00390625, 1.0),
            pair(-1.01171875, 1.0),
            pair(1.0, -1.00390625),
            pair(-1.00390625, -1.00390625),
            pair(-128.5, 1.00390625),
            pair(-129.5, -1.01171875),
        ],
        # Neighboring representables: 1 FP32 ULP below, at, and above BF16 half-way.
        "neighboring_representables": [
            (0x3F807FFF, f32_to_bits(1.0)),  # just below tie -> 1.0
            (0x3F808000, f32_to_bits(1.0)),  # exact tie -> 1.0078125
            (0x3F808001, f32_to_bits(1.0)),  # just above tie -> 1.0078125
            (f32_to_bits(1.0), 0x3F807FFF),
            (f32_to_bits(1.0), 0x3F808000),
            (f32_to_bits(1.0), 0x3F808001),
            (0xBF807FFF, f32_to_bits(1.0)),
            (0xBF808000, f32_to_bits(1.0)),
            (0xBF808001, f32_to_bits(1.0)),
        ],
        # Sign changes across all quadrants.
        "sign_changes": [
            pair(2.25, 3.5),
            pair(-2.25, 3.5),
            pair(2.25, -3.5),
            pair(-2.25, -3.5),
            pair(-0.501953125, 2.0),  # negative tie in [0.5, 1.0)
            pair(0.501953125, -2.0),
        ],
        # Signed zero (+0.0 and -0.0).
        "signed_zero": [
            (0x00000000, f32_to_bits(3.0)),
            (0x80000000, f32_to_bits(3.0)),
            (f32_to_bits(3.0), 0x80000000),
            (f32_to_bits(-3.0), 0x00000000),
            (0x80000000, 0x80000000),
            (0x00000000, 0x80000000),
        ],
        # Small normal magnitudes staying strictly in normal BF16 domain.
        "small_normal_magnitudes": [
            pair(2.0**-60, 2.0**-60),
            pair(-(2.0**-60), 2.0**-60),
            pair(2.0**-63, 2.0**-63),  # exact 2^-126 normal minimum
            pair((1.0 + 2.0**-8) * (2.0**-60), 2.0**-60),
        ],
        # Rounding carry across mantissa and exponent boundaries.
        "rounding_carry": [
            (0x3F7F8000, f32_to_bits(1.0)),  # 1 - 2^-9 -> carries to 1.0 (0x3F80)
            (0x3FFF8000, f32_to_bits(1.0)),  # 2 - 2^-8 -> carries to 2.0 (0x4000)
            (0xBF7F8000, f32_to_bits(1.0)),  # carries to -1.0 (0xBF80)
            (f32_to_bits(1.0), 0x3FFF8000),
            (0x3FFF8000, 0x3FFF8000),  # 2.0 * 2.0 = 4.0
        ],
        # Large finite values staying strictly below BF16 overflow.
        "large_finite": [
            pair(2.0**60, 2.0**60),
            pair(-(2.0**60), 2.0**60),
            pair((1.0 + 2.0**-8) * (2.0**60), 2.0**60),
            (0x7F7E8000, f32_to_bits(1.0)),  # carries to BF16 max finite 0x7F7F
        ],
        # Final-product RNE tie witnesses: exact BF16 inputs whose product is an
        # exact BF16 half-way tie (separates Stage-4 RNE from Stage-3 RNA!).
        "final_product_rne_ties": [
            # 1.5 * (1 + 3*2^-7) = 1 + 68*2^-7 + 2^-8 -> even floor 68 -> RNE 0x3FC4 (RNA 0x3FC5!)
            (0x3FC00000, 0x3F830000),
            # 1.5 * (1 + 1*2^-7) = 1 + 65*2^-7 + 2^-8 -> odd floor 65 -> RNE 0x3FC2
            (0x3FC00000, 0x3F810000),
            # Negative counterparts:
            (0xBFC00000, 0x3F830000),  # RNE 0xBFC4 vs RNA 0xBFC5
            (0xBFC00000, 0x3F810000),  # RNE 0xBFC2
            # Near-tie below and above exact product half-way (tests truncation vs rounding):
            (
                0x3F810000,
                0x3F810000,
            ),  # (1+2^-7)^2 = 1 + 2*2^-7 + 2^-14 -> rounds down to 0x3F82
            (
                0x3F8F0000,
                0x3F8F0000,
            ),  # (1+15*2^-7)^2 -> fractional > 1/2 -> rounds UP (truncation fails!)
        ],
        # Correlated / repeated deterministic families.
        "correlated_repeated": [
            pair(1.00390625, 1.00390625),
            pair(1.00390625, 1.00390625),
            pair(-1.00390625, 1.00390625),
            pair(1.5 + 2.0**-8, 1.5 + 2.0**-8),
            pair(3.125, -3.125),
            pair(0.75 + 2.0**-9, 0.75 + 2.0**-9),
        ],
        # Out-of-domain: nonfinite inputs (must be explicitly classified, never accepted).
        "out_of_domain_nonfinite": [
            (0x7F800000, f32_to_bits(1.0)),  # +Inf
            (0xFF800000, f32_to_bits(1.0)),  # -Inf
            (0x7FC00000, f32_to_bits(1.0)),  # qNaN
            (0x7FFFFFFF, f32_to_bits(1.0)),  # max NaN (wraps under +0x8000!)
            (f32_to_bits(1.0), 0x7FC00000),
        ],
        # Out-of-domain: intermediate cast or product overflow.
        "out_of_domain_overflow": [
            (0x7F7F8000, f32_to_bits(1.0)),  # intermediate RNA carry into Inf
            (0xFF7F8000, f32_to_bits(1.0)),  # negative intermediate carry into -Inf
            pair(2.0**70, 2.0**70),  # product 2^140 overflows BF16
            pair(-(2.0**70), 2.0**70),
        ],
        # Inconclusive: subnormal inputs or underflow into subnormal (FTZ unmodeled).
        "inconclusive_subnormal_underflow": [
            (0x00000001, f32_to_bits(1.0)),  # FP32 smallest subnormal
            (0x007FFFFF, f32_to_bits(1.0)),  # FP32 largest subnormal
            (f32_to_bits(1.0), 0x00400000),  # subnormal up
            pair(2.0**-70, 2.0**-70),  # normal inputs underflowing to 2^-140
        ],
    }


def _file_sha256(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def evaluate_epilogue_against_oracle(
    epilogue_fn: Callable[[Any, Any], Any],
    *,
    fixtures: dict[str, list[tuple[int, int]]] | None = None,
    require_hardware_zero_sign: bool = False,
) -> dict[str, Any]:
    """Evaluate a tensor epilogue callable against the independent rational oracle.

    `epilogue_fn(gate_act_f32_tensor, up_f32_tensor)` must return a `torch.bfloat16`
    tensor of the same length. Non-BF16 return dtype is recorded as a violation.
    """
    import torch

    if fixtures is None:
        fixtures = build_discriminating_fixtures()

    domain_counts: dict[str, int] = {
        DOMAIN_IN_DOMAIN: 0,
        DOMAIN_NONFINITE: 0,
        DOMAIN_OVERFLOW: 0,
        DOMAIN_SUBNORMAL_UNMODELED: 0,
    }
    in_domain_checked = 0
    mismatches: list[dict[str, Any]] = []
    zero_sign_differences: list[dict[str, Any]] = []
    dtype_errors: list[str] = []

    for family_name, pairs in fixtures.items():
        g_bits_list = [p[0] for p in pairs]
        u_bits_list = [p[1] for p in pairs]
        oracle_results = [
            epilogue_oracle_element(gb, ub)
            for gb, ub in zip(g_bits_list, u_bits_list, strict=True)
        ]

        in_domain_indices: list[int] = []
        for idx, res in enumerate(oracle_results):
            domain_counts[res.domain_status] = (
                domain_counts.get(res.domain_status, 0) + 1
            )
            if res.domain_status == DOMAIN_IN_DOMAIN:
                in_domain_indices.append(idx)

        if not in_domain_indices:
            continue

        g_in = torch.tensor(
            [g_bits_list[i] for i in in_domain_indices], dtype=torch.uint32
        ).view(torch.float32)
        u_in = torch.tensor(
            [u_bits_list[i] for i in in_domain_indices], dtype=torch.uint32
        ).view(torch.float32)

        actual = epilogue_fn(g_in, u_in)
        if actual.dtype != torch.bfloat16:
            dtype_errors.append(
                f"{family_name}: expected torch.bfloat16 output, got {actual.dtype}"
            )
            continue

        actual_bf16_bits = [
            int(v) for v in actual.contiguous().view(torch.uint16).tolist()
        ]
        for local_i, orig_i in enumerate(in_domain_indices):
            in_domain_checked += 1
            res = oracle_results[orig_i]
            assert res.expected_bf16_bits is not None
            got_bits = actual_bf16_bits[local_i]
            if got_bits != res.expected_bf16_bits:
                # Check if the ONLY discrepancy is IEEE -0.0 vs hardware +0.0 canonicalization.
                if (
                    res.exact_product_rational == 0
                    and got_bits == res.ieee_signed_zero_bf16_bits
                    and not require_hardware_zero_sign
                ):
                    zero_sign_differences.append({
                        "family": family_name,
                        "index": orig_i,
                        "gate_act_f32_hex": f"0x{res.gate_act_f32_bits:08x}",
                        "up_f32_hex": f"0x{res.up_f32_bits:08x}",
                        "hardware_expected_bf16_hex": f"0x{res.expected_bf16_bits:04x}",
                        "observed_bf16_hex": f"0x{got_bits:04x}",
                        "note": "IEEE signed-zero (-0.0) vs ckernel_sfpu_binary.h:163 +0.0 canonicalization",
                    })
                else:
                    mismatches.append({
                        "family": family_name,
                        "index": orig_i,
                        "gate_act_f32_hex": f"0x{res.gate_act_f32_bits:08x}",
                        "up_f32_hex": f"0x{res.up_f32_bits:08x}",
                        "expected_bf16_hex": f"0x{res.expected_bf16_bits:04x}",
                        "observed_bf16_hex": f"0x{got_bits:04x}",
                    })

    passed = not dtype_errors and not mismatches
    return {
        "scope": ORACLE_SCOPE,
        "passed": passed,
        "in_domain_checked": in_domain_checked,
        "dtype_errors": dtype_errors,
        "mismatch_count": len(mismatches),
        "mismatches": mismatches[:10],
        "zero_sign_canonicalization_differences": zero_sign_differences,
        "domain_coverage": domain_counts,
    }


def _bf16_ties_away_torch(x):
    """Helper used only inside mutation harnesses to construct explicit mutants."""
    import torch

    bits = x.float().contiguous().view(torch.int32)
    return ((bits + 0x8000) & -65536).view(torch.float32).to(torch.bfloat16)


def run_mutation_controls() -> dict[str, Any]:
    """Run all required epilogue mutation controls and the common-mode control."""
    import torch

    fixtures = build_discriminating_fixtures()

    def hardware_exact_epilogue(g_act, u):
        g_r = _bf16_ties_away_torch(g_act)
        u_r = _bf16_ties_away_torch(u)
        prod = (g_r.float() * u_r.float()).to(torch.bfloat16)
        zero_mask = (g_r == 0) | (u_r == 0)
        return torch.where(zero_mask, torch.zeros_like(prod), prod)

    # Positive check: hardware-specified epilogue passes even with strict +0.0 zero sign.
    hw_positive = evaluate_epilogue_against_oracle(
        hardware_exact_epilogue,
        fixtures=fixtures,
        require_hardware_zero_sign=True,
    )
    if not hw_positive["passed"]:
        raise AssertionError(
            f"hardware-exact epilogue failed independent oracle: {hw_positive}"
        )

    mutants: dict[str, Callable[[Any, Any], Any]] = {
        # 1. Removed intermediate gate rounding.
        "removed_gate_rounding": lambda g, u: (
            g.float() * _bf16_ties_away_torch(u).float()
        ).to(torch.bfloat16),
        # 2. Removed intermediate up rounding.
        "removed_up_rounding": lambda g, u: (
            _bf16_ties_away_torch(g).float() * u.float()
        ).to(torch.bfloat16),
        # 3. Ties-even substituted for intermediate ties-away.
        "ties_even_intermediate": lambda g, u: (
            g.to(torch.bfloat16).float() * u.to(torch.bfloat16).float()
        ).to(torch.bfloat16),
        # 4. Truncated rather than rounded final store.
        "truncated_final_store": lambda g, u: (
            (
                (_bf16_ties_away_torch(g).float() * _bf16_ties_away_torch(u).float())
                .contiguous()
                .view(torch.int32)
                & -65536
            )
            .view(torch.float32)
            .to(torch.bfloat16)
        ),
        # 5. FP32 product retained across the required BF16 output boundary.
        "fp32_product_retained": lambda g, u: (
            _bf16_ties_away_torch(g).float() * _bf16_ties_away_torch(u).float()
        ),
        # 6. Wrong final product tie rule (ties-away instead of RNE).
        "wrong_final_store_ties_away": lambda g, u: _bf16_ties_away_torch(
            _bf16_ties_away_torch(g).float() * _bf16_ties_away_torch(u).float()
        ),
        # 7. Incorrect final store (corrupted LSB on non-zero outputs).
        "wrong_final_store_corrupted_lsb": lambda g, u: (
            (_bf16_ties_away_torch(g).float() * _bf16_ties_away_torch(u).float())
            .to(torch.bfloat16)
            .contiguous()
            .view(torch.uint16)
            ^ torch.tensor(1, dtype=torch.uint16)
        ).view(torch.bfloat16),
    }

    mutant_reports: dict[str, dict[str, Any]] = {}
    for name, fn in mutants.items():
        res = evaluate_epilogue_against_oracle(
            fn, fixtures=fixtures, require_hardware_zero_sign=False
        )
        detected = not res["passed"]
        if not detected:
            raise AssertionError(
                f"mutation control failed to detect injected fault {name!r}"
            )
        witness = (
            res["mismatches"][0]
            if res["mismatches"]
            else {"dtype_error": res["dtype_errors"][0]}
        )
        mutant_reports[name] = {
            "detected": True,
            "mismatch_count": res["mismatch_count"],
            "dtype_error_count": len(res["dtype_errors"]),
            "witness": witness,
        }

    # Common-mode control: baseline and candidate share the SAME wrong arithmetic
    # (ties-even intermediate casts instead of ties-away). Pairwise torch.equal
    # passes 100%, while the independent rational oracle rejects both!
    def shared_wrong_arithmetic(g, u):
        return (g.to(torch.bfloat16).float() * u.to(torch.bfloat16).float()).to(
            torch.bfloat16
        )

    witness_g = torch.tensor([1.00390625, -1.00390625, 128.5], dtype=torch.float32)
    witness_u = torch.tensor([1.0, 1.0, 1.00390625], dtype=torch.float32)
    baseline_wrong = shared_wrong_arithmetic(witness_g, witness_u)
    candidate_wrong = shared_wrong_arithmetic(witness_g, witness_u)
    pairwise_equal = bool(torch.equal(baseline_wrong, candidate_wrong))
    oracle_check = evaluate_epilogue_against_oracle(
        shared_wrong_arithmetic,
        fixtures={
            "common_mode_witness": [
                (f32_to_bits(float(g)), f32_to_bits(float(u)))
                for g, u in zip(witness_g.tolist(), witness_u.tolist(), strict=True)
            ]
        },
        require_hardware_zero_sign=False,
    )
    if not pairwise_equal:
        raise AssertionError("common-mode baseline and candidate unexpectedly differed")
    if oracle_check["passed"]:
        raise AssertionError("common-mode wrong arithmetic escaped independent oracle")

    return {
        "hardware_exact_positive": {
            "passed": hw_positive["passed"],
            "in_domain_checked": hw_positive["in_domain_checked"],
        },
        "mutants": mutant_reports,
        "common_mode_control": {
            "pairwise_torch_equal_passed": pairwise_equal,
            "independent_oracle_rejected": not oracle_check["passed"],
            "mismatch_count": oracle_check["mismatch_count"],
            "witness": oracle_check["mismatches"][0],
        },
    }


def verify_rounded_reference_epilogue(
    rounded_reference_fn: Callable[[Any, Any], Any],
) -> dict[str, Any]:
    """Cross-check production rounded_reference(gate, up) against the independent oracle.

    Because rounded_reference applies PyTorch SiLU to `gate` before the epilogue,
    we pass `(torch.nn.functional.silu(gate.float()), up)` to the independent
    rational epilogue oracle and compare bit-for-bit against `rounded_reference(gate, up)`.
    This verifies the epilogue (Stages 3a/3b/4/5) without assuming anything about SiLU.
    """
    import torch

    # Deterministic families exercising up across all ties/neighbors/carries/RNE-product ties
    # and gate across negative, zero, small, and large (silu(x)==x) regimes.
    gate_cases: list[int] = [
        f32_to_bits(24.0),  # 1.5 * 2^4 (exact under silu)
        f32_to_bits(24.0),  # paired with 1 + 3*2^-7 -> exact RNE even-LSB tie!
        f32_to_bits(24.0),  # paired with 1 + 1*2^-7 -> exact RNE odd-LSB tie!
        f32_to_bits(32.0),  # exact power of 2
        f32_to_bits(32.0),
        f32_to_bits(32.0),
        f32_to_bits(32.0),
        f32_to_bits(32.0),
        f32_to_bits(32.0),
        f32_to_bits(128.5),  # exact positive half-way tie in gate
        f32_to_bits(129.5),  # exact positive half-way tie (odd LSB) in gate
        0x42008000,  # 32 + 2^-3 (gate positive tie, even LSB)
        0x42007FFF,  # 1 FP32 ULP below tie
        0x42008001,  # 1 FP32 ULP above tie
        0x41FF8000,  # rounding carry in gate -> 32.0
        f32_to_bits(-1.25),  # negative gate -> negative silu output
        f32_to_bits(-0.5),
        f32_to_bits(-3.0),
        f32_to_bits(-128.0),  # saturates to -0.0 / 0.0 under silu
        0x00000000,  # +0.0
        0x80000000,  # -0.0
        f32_to_bits(2.0**60),  # large finite
    ]
    up_cases: list[int] = [
        0x3F830000,  # 1 + 3*2^-7 (final-product RNE tie, even LSB -> keeps even)
        0x3F810000,  # 1 + 1*2^-7 (final-product RNE tie, odd LSB -> rounds up)
        0xBF830000,  # negative final-product RNE tie
        f32_to_bits(
            1.00390625
        ),  # positive half-way tie in up (even LSB -> RNA rounds away)
        f32_to_bits(-1.00390625),  # negative half-way tie in up (RNA rounds away)
        0x3F807FFF,  # neighboring below tie
        0x3F808001,  # neighboring above tie
        0x3F7F8000,  # rounding carry -> 1.0
        0x3FFF8000,  # rounding carry -> 2.0
        f32_to_bits(1.00390625),
        f32_to_bits(1.01171875),
        f32_to_bits(1.0),
        f32_to_bits(1.0),
        f32_to_bits(1.0),
        f32_to_bits(1.0),
        f32_to_bits(1.00390625),
        f32_to_bits(-1.00390625),
        f32_to_bits(3.5),
        f32_to_bits(3.0),
        f32_to_bits(-3.0),
        f32_to_bits(3.0),
        f32_to_bits(2.0**50),
    ]
    assert len(gate_cases) == len(up_cases)

    g_tensor = torch.tensor(gate_cases, dtype=torch.uint32).view(torch.float32)
    u_tensor = torch.tensor(up_cases, dtype=torch.uint32).view(torch.float32)
    actual = rounded_reference_fn(g_tensor, u_tensor)
    if actual.dtype != torch.bfloat16:
        raise AssertionError(
            f"rounded_reference returned {actual.dtype}, expected bfloat16"
        )

    gate_act = torch.nn.functional.silu(g_tensor.float())
    gate_act_bits = [int(v) for v in gate_act.contiguous().view(torch.uint32).tolist()]
    actual_bits = [int(v) for v in actual.contiguous().view(torch.uint16).tolist()]

    checked = 0
    mismatches: list[dict[str, Any]] = []
    zero_sign_diffs: list[dict[str, Any]] = []
    out_of_domain_skipped = 0
    for idx, (ga_bits, ub, got) in enumerate(
        zip(gate_act_bits, up_cases, actual_bits, strict=True)
    ):
        res = epilogue_oracle_element(ga_bits, ub)
        if res.domain_status != DOMAIN_IN_DOMAIN:
            out_of_domain_skipped += 1
            continue
        checked += 1
        assert res.expected_bf16_bits is not None
        if got != res.expected_bf16_bits:
            if (
                res.exact_product_rational == 0
                and got == res.ieee_signed_zero_bf16_bits
            ):
                zero_sign_diffs.append({
                    "index": idx,
                    "hardware_expected_bf16_hex": f"0x{res.expected_bf16_bits:04x}",
                    "observed_bf16_hex": f"0x{got:04x}",
                })
            else:
                mismatches.append({
                    "index": idx,
                    "gate_act_f32_hex": f"0x{ga_bits:08x}",
                    "up_f32_hex": f"0x{ub:08x}",
                    "expected_bf16_hex": f"0x{res.expected_bf16_bits:04x}",
                    "observed_bf16_hex": f"0x{got:04x}",
                })

    if mismatches:
        raise AssertionError(
            f"production rounded_reference disagreed with independent epilogue oracle: {mismatches}"
        )
    return {
        "passed": True,
        "checked_elements": checked,
        "out_of_domain_skipped": out_of_domain_skipped,
        "zero_sign_canonicalization_differences": zero_sign_diffs,
    }


def _find_precision_plan(module_dir: Path) -> Path | None:
    candidates = [
        module_dir.parents[1] / PRECISION_PLAN_RELATIVE_PATH,
        module_dir / "rounded-register-swiglu-precision-plan.md",
    ]
    for parent in module_dir.parents:
        candidates.append(
            parent / "share" / "qwen36" / "rounded-register-swiglu-precision-plan.md"
        )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def run_epilogue_qualification(
    rounded_reference_fn: Callable[[Any, Any], Any],
    *,
    module_dir: Path,
) -> dict[str, Any]:
    """Run independent epilogue oracle, domain checks, and mutation controls."""
    started = time.perf_counter()
    fixtures = build_discriminating_fixtures()

    # 1. Verify production rounded_reference epilogue against independent rational oracle.
    prod_ref_report = verify_rounded_reference_epilogue(rounded_reference_fn)

    # 2. Verify all 13 discriminating fixture families (including domain classification).
    import torch

    def hardware_model(g_act, u):
        g_r = _bf16_ties_away_torch(g_act)
        u_r = _bf16_ties_away_torch(u)
        prod = (g_r.float() * u_r.float()).to(torch.bfloat16)
        zero_mask = (g_r == 0) | (u_r == 0)
        return torch.where(zero_mask, torch.zeros_like(prod), prod)

    oracle_eval = evaluate_epilogue_against_oracle(
        hardware_model,
        fixtures=fixtures,
        require_hardware_zero_sign=True,
    )
    if not oracle_eval["passed"]:
        raise AssertionError(
            f"hardware-specified epilogue failed independent oracle: {oracle_eval}"
        )

    # Confirm out-of-domain / inconclusive fixtures were explicitly classified
    # (never dropped or passed as in-domain).
    cov = oracle_eval["domain_coverage"]
    if (
        cov.get(DOMAIN_NONFINITE, 0) < 1
        or cov.get(DOMAIN_OVERFLOW, 0) < 1
        or cov.get(DOMAIN_SUBNORMAL_UNMODELED, 0) < 1
    ):
        raise AssertionError(f"incomplete domain coverage classification: {cov}")

    # 3. Verify all mutation controls + common-mode control.
    mutation_report = run_mutation_controls()
    elapsed = time.perf_counter() - started

    plan_path = _find_precision_plan(module_dir)
    return {
        "independent_epilogue_oracle": {
            "scope": ORACLE_SCOPE,
            "passed": oracle_eval["passed"] and prod_ref_report["passed"],
            "discriminating_in_domain_checked": oracle_eval["in_domain_checked"],
            "production_rounded_reference_checked": prod_ref_report["checked_elements"],
            "production_zero_sign_differences": prod_ref_report[
                "zero_sign_canonicalization_differences"
            ],
            "mismatch_count": oracle_eval["mismatch_count"],
        },
        "domain_coverage": cov,
        "mutation_controls": mutation_report["mutants"],
        "common_mode_control": mutation_report["common_mode_control"],
        "unresolved_assumptions": [
            "Blackhole FPU LoFi mantissa truncation width and K-tile accumulation order are hardware-owned/unknown.",
            "Blackhole SFPU/packer subnormal FTZ behavior is hardware-owned/unknown; subnormal inputs/underflows are classified inconclusive.",
            "SFPU SiLU (_sfpu_exp_accurate_<true> + sfpu_reciprocal_iter<2>) ULP envelope on silicon is device-owned/unmodeled.",
        ],
        "blocked_full_kernel_stages": [
            "stage_0_bf4_block_quantization",
            "stage_1_lofi_fp32_matmul_accumulation",
            "stage_2_sfpu_silu_activation",
        ],
        "bindings": {
            "precision_plan": PRECISION_PLAN_RELATIVE_PATH,
            "precision_plan_sha256": _file_sha256(plan_path) if plan_path else None,
            "pinned_tt_metal_version": PINNED_TT_METAL_VERSION,
            "compute_cpp_sha256": _file_sha256(
                module_dir / "rounded_register_kernels" / "compute.cpp"
            ),
            "rounded_register_swiglu_sha256": _file_sha256(
                module_dir / "rounded_register_swiglu.py"
            ),
            "oracle_sha256": _file_sha256(Path(__file__)),
        },
        "reference_check_wall_seconds": elapsed,
        "cost_accounting_note": "CPU independent oracle and mutation-control wall time reported separately from device kernel timing.",
    }
