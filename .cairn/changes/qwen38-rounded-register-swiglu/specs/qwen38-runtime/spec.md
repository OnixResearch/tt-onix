## ADDED Requirements

### Requirement: Rounded register SwiGLU remains a precision-preserving decode candidate

r[tenstorrent.qwen38.rounded_register_swiglu]

The Qwen candidate MUST remain default-off and apply only to supported B1/B8 decode or verify geometry, preserving prefill, down projection, collectives and the independent fused-gate/up admission flag. Its gate/up BF16 boundaries MUST match native pack ties-away while its BF16 product remains ties-even. Qualification MUST preserve exact component equality, content-bound original-six-passage B1 retention and explicit B8 continuation bounds; slower component timings MUST NOT authorize performance or serving promotion.

#### Scenario: Supported decode uses the register epilogue

- GIVEN the actual TP2-local BF4 gate/up weights and original LoFi/FP32/approximate math policy
- WHEN the opt-in executes B1 or B8 decode repeatedly
- THEN the BF16 product is exactly equal to the original fused-SiLU gate plus up/product route, while unsupported requested shapes fail and prefill stays on the original path.

#### Scenario: The original simulator functionality is unavailable

- GIVEN the unchanged physical candidate kernel and original TTsim 1.10.6 model
- WHEN TTsim rejects Disable_pack_zero_flags after JIT compilation
- THEN the receipt preserves that dated blocked attempt without relabeling it as a successful simulation or introducing a simulator-only kernel workaround.

#### Scenario: The bounded simulator model resolves the component blocker

- GIVEN source-pinned TTsim 1.10.6 with the Blackhole dense-L1 PACR model extension and the unchanged installed Qwen/TT-Metal 0.77.0 binaries
- WHEN the B1/B8 TP2-local component comparisons execute on one simulated chip
- THEN both baseline and candidate match exactly in BF16, the receipt keeps the original blocked attempt separately, native TTsim 1.10.9 and the separate GDN limitation remain unchanged, and no full-model, multi-chip, silicon-performance or serving acceptance follows.

#### Scenario: Numerical checks pass but component timing regresses

- GIVEN physical B1/B8 component equality, content-bound B1 retention and bounded B8 continuation
- WHEN installed component timings are slower than baseline
- THEN the numerical evidence is retained, B8 retention remains unproven, and performance promotion, serving admission and default changes remain unauthorized.
