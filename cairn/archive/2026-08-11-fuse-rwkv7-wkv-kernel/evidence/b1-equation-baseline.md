# Fused WKV kernel — B1 equation and shape baseline

Task B1. This document captures the accepted decomposed WKV equation, the
tensor shapes, the precision boundaries, the launch profile, and the token
streams, so the fused operator has an exact reference to preserve.

Sources read for this baseline:

- RWKV-7 Goose paper, arXiv:2503.14456v2, Section 4.1 Time Mixing and
  Appendix G pseudocode.
- Upstream reference implementation `RWKV-LM/RWKV-v5/cuda/wkv7_cuda.cu`
  (the canonical token-at-a-time fused forward kernel).
- The packaged runtime `pkgs/rwkv7-p150x2/src/main.cpp`
  (`prepare_time_mix`, `wkv_step`, `finish_attention`).
- Blackhole ISA docs (`tt-isa-documentation/BlackholeA0`) and the
  packaged tt-metal 0.74.0 LLK compute API headers.

## Accepted equation (per head, per token)

The recurrent state `S` is `D x D` per head, `D = 64` (head size). The
paper (Section 4.1, Eq. 17-22) defines, per head:

```
S_t = S_{t-1} * diag(w_t)  -  kappa_hat_t (a_t . kappa_hat_t)^T  +  v_t k_tilde_t^T
```

The runtime's `wkv_step` implements the equivalent decomposed form with
`kk = normalize_heads(k * k_k)` (the normalized removal key `kappa_hat`),
`a = -kk`, `b = kk * a_gate`, and `k_tilde = k * (1 + (a_gate - 1) * k_a)`:

```
decay       = S * w_rows                       (elementwise, w per channel)
rank_update = (S . a_rows) b_rows              (two matmuls)
outer       = v_state_rows k_state_rows^T      (outer product)
S_next      = decay + rank_update + outer
output      = S_next . r_rows                  (readout)
```

This matches the canonical fused reference in `wkv7_cuda.cu`:

```
for j in C:
    s[j] = s[j] * w[j] + sa * b[j] + k[j] * v[j]
    y   += s[j] * q[j]
```

where `sa = sum_j a[j] * s[j]` (the rank-update pre-product), `q = r`,
`w` is the decay, `k`/`v` are the BF16 key/value, and `a`/`b` are the
rank-update vectors.

## Tensor shapes

All shapes are per decode step with `batch` live request slots.

| Tensor | Shape | Dtype | Source |
| --- | --- | --- | --- |
| recurrent state `S` | `[batch, heads, D, D]` = `[b,12,64,64]` | FP32 | device-resident |
| `r, w, k, v, a, b` rows | `[batch, heads, 1, D]` = `[b,12,1,64]` | BF16 (state row cast to FP32) | `prepare_time_mix` |
| `kk` (normalized removal key) | `[batch, heads, 1, D]` | BF16 | `normalize_heads` |
| `S_next` | `[batch, heads, D, D]` | FP32 | `wkv_step` |
| `output_for_hidden` | `[batch, heads, D]` | BF16 | `wkv_step` |

`heads = 12`, `D = 64`, `kHiddenSize = 768 = 12 * 64`.

## Precision boundaries

- Recurrent state `S` and `S_next` are FP32 (device-resident).
- Model vectors `r, w, k, v, a, b` enter as BF16 and are cast to FP32 only
  for the state matmul (`typecast-recurrence-fp32`).
- The attention output `output_for_hidden` is BF16 written directly by the
  FP32 WKV matmul (`op_matmul_transpose_right_bf16`), avoiding a separate
  recurrent-output typecast.
- The removal key `kk` stays BF16; its normalization divides by the L2 norm
  of each head row.

## Launch profile (current decomposed path, per layer per step)

The decomposed `wkv_step` dispatches these TT-NN operations per layer per
token:

1. reshape `a, b, w` to state rows;
2. reshape/cast `v, k, r` to state rows and FP32;
3. `matmul_transpose_right(S, a)` -> `S . a`;
4. `matmul(S.a, b)` -> rank update;
5. `multiply(S, w)` -> decay;
6. `matmul_transpose_left(v, k)` -> outer;
7. `add(decay, rank_update)`;
8. `add(..., outer)` -> `S_next`;
9. `matmul_transpose_right(S_next, r)` -> output (BF16).

The fused kernel folds steps 3-9 into one device program: decay, both
rank-update products, the outer, the state write, and the BF16 readout.

## Token streams

The exact generated-token streams are pinned by the FP32 CPU reference used
by the admitted evidence fixtures. The fused path must reproduce them bit
for bit at the greedy-selection boundary. See
`pkgs/rwkv7-p150x2-evidence/fixtures/` for the retained streams.

## Non-claims

This baseline does not claim the fused path is faster. It records the
equation and shapes the fused kernel must preserve. Launch reduction alone
is not completion evidence (see the change's `no_gain` scenario).