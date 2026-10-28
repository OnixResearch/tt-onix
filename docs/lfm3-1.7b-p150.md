# LFM3-1.7B on one Blackhole P150

Liquid AI released LFM3 on 2026-10-27. `LiquidAI/LFM3-1.7B` is the text member
of the family: a 1.7B-parameter tri-hybrid model with a 128K vocabulary and a
256K-token context window. This note records the TT-NN bring-up in `pkgs/lfm3`,
the measured accuracy against a CPU fp32 oracle, the decode/prefill throughput
on one card, and the precision floor of the packaged TT-Metal runtime.

## Architecture (checkpoint `config.json` + `modeling_lfm3.py`)

- 24 decoder layers: 13 gated short convolutions (kernel 7), 5 sliding-window
  attention layers (window 256, full RoPE, theta 1e4), 6 global attention
  layers (partial RoPE, `partial_rotary_factor = 0.5`, theta 1e6).
- Hidden 2048, FFN 7680 (SwiGLU), 16 query / 4 KV heads, head_dim 128.
- Per-head sigmoid output gate (`gate_proj` is `[16, 2048]`, one scalar per
  head), QK RMSNorm, sandwich RMSNorm around both residual branches.
- Embedding tied to the LM head; 128000 rows; `norm_eps = 1e-5`.
- The convolution is a depthwise causal width-7 filter with `conv_L_cache = 7`;
  the recurrent window for decode is the last 6 pre-conv rows.

The runner mirrors `modeling_lfm3.py` operation for operation. Two details are
easy to get wrong and are covered by the gate: the partial-RoPE tail must be
`cos = 1, sin = 0` for the unrotated half, and the convolution window is
`y[t] = sum_k w[k] * x[t-6+k]` (left-padded, not centered).

## Running

```sh
nix build .#lfm3 --option builders ""
out=$(nix build .#lfm3 --no-link --print-out-paths --option builders "")

# Checkpoint: Hugging Face files in one directory (config.json,
# model.safetensors, tokenizer.json). The default is ~/.cache/lfm3 or $LFM3_CHECKPOINT_DIR.
"$out/bin/lfm3-run"  --checkpoint-dir DIR --tokens 32 --dtype float32
"$out/bin/lfm3-run"  --checkpoint-dir DIR --tokens 32 --dtype bfloat16   # fast path
"$out/bin/lfm3-run"  --checkpoint-dir DIR --tokens 32 --dtype float32 --decode-attention fp32
"$out/bin/lfm3-check" --checkpoint-dir DIR --dtype float32               # device vs oracle
```

The first device run compiles the TT-Metal kernels for these shapes and takes
about 20 s for prefill; later runs reuse nothing on disk (the JIT cache is
in-process in this build). The device entry points serve a writable
``TT_METAL_HOME`` from ``${XDG_CACHE_HOME:-~/.cache}/lfm3/tt-metal-runtime``
(a symlink tree of the packaged overlay) on first use.

The CPU oracle is not part of the derivation: the checkpoint's remote code
declares ``transformers_version = 5.12.1`` and imports
``transformers.integrations.use_kernel_func_from_hub_with_fallback``, which the
nixpkgs transformers does not export. Run it in a matching virtualenv:

```sh
python pkgs/lfm3/oracle/reference.py --checkpoint-dir DIR --tokens 32
```

The raw `tokenizers` library does not match the checkpoint's non-special
` thinking` added token (`tokenizer.json` id 124901) on its own, so the runner
appends that id after encoding the chat prompt; `lfm3-reference` gets it from
the transformers tokenizer for free, and both produce the same 17 ids.

The oracle writes `reference.npz` (embedding, all 24 layer outputs, the final
norm, prefill logits, greedy tokens, per-step logits). `lfm3-check` replays the
same prompt on the card and prints per-layer max/mean absolute error and PCC
plus the decode token comparison.

## Measured evidence

Card 0 of the local P150x2 host, TT-Metal 0.77.0, fast dispatch. Prompt
`What is C. elegans?` with the checkpoint chat template (17 tokens, ending in
` thinking`).

| quantity | fp32 mode | bf16 mode |
| --- | --- | --- |
| greedy tokens vs CPU fp32 oracle (32 generated) | 32/32 | 31/32 |
| prefill, 17-token prompt (warm) | 55-70 tok/s | 75 tok/s |
| decode (default bf16 SDPA kernel) | 25-26 tok/s | 22.6-24 tok/s |
| decode (`--decode-attention fp32`) | 5.8 tok/s | n/a |
| prefill logits vs oracle | PCC 0.99927, max abs 0.54 | - |
| hidden states, embedding to layer 12 | PCC >= 0.9998 | - |
| worst captured hidden state (final norm) | PCC 0.9982, max abs 0.81 | - |

430-token prompt (the question repeated 60 times, crossing the 256-token
sliding window):

| quantity | fp32 mode | bf16 mode |
| --- | --- | --- |
| greedy tokens vs CPU fp32 oracle (32 generated) | 32/32 | 0/32 |
| greedy tokens, `--decode-attention fp32` | 32/32 | n/a |
| prefill (warm) | 918 tok/s | 1119 tok/s |
| decode (default bf16 SDPA kernel) | 26.1 tok/s | 20.3 tok/s |
| prefill logits vs oracle | PCC 0.99932, max abs 0.63 | - |
| hidden state, layer 23 / final norm | PCC 0.928 / 0.939 | - |

The wide-cache decode option reproduces the same 32 tokens at 4.5x the
per-token cost; on this prompt the bf16 SDPA decode kernel is not what limits
agreement.

The 430-token prefill logits stay within 0.63 absolute of the oracle and pick
the same argmax, while the last hidden state has drifted to PCC 0.93. The
greedy stream is insensitive to that drift at this prompt; treat the hidden
PCC as the honest long-context accuracy number, not the token match.

The 32-token oracle stream is
`2083, 1094, 310, 5141, 278, 3451, 34, 496, 2992, 355, 340, 22, 13729, 497,
8022, 969, 355, 267, 90139, 15789, 3451, 22, 340, 22, 13729, 497, 355, 267,
1931, 302, 58330, 1083` ("We need to answer the question: ..."). The bf16 run
first diverges at index 18.

## What the optimization changed

- One fused QKV projection per attention layer (`[q | k | v]` rows, split on
  the device) and one fused gate/up FFN projection (`w13`, split on the
  device). The per-head attention gate stays separate: its weight is
  `[16, 2048]`, so folding it into the QKV matmul would make the output width
  non-tile-aligned (3088 columns).
- `ttnn.experimental.rotary_embedding_hf` replaces the eight-op manual
  rotate-half; the partial-RoPE layers keep their `cos = 1, sin = 0` tail.
- The decode convolution is one `concat` + `multiply` + `sum` over the six
  cached rows and the new row (three dispatch ops instead of twelve).
- The conv window is kept as six row-major device rows (oldest first) that the
  Python step rotates; the prefill path accumulates seven shifted taps.

<!-- r[impl tenstorrent.lfm3.precision.disclosure] -->
<!-- r[verify tenstorrent.lfm3.precision.disclosure.matmul] -->
<!-- r[verify tenstorrent.lfm3.precision.disclosure.bfloat16] -->
## Precision floor (measured, not assumed)

Every variant of `ttnn.linear` was measured against an exact product of the
same inputs on the same card (M = 32, K = N = 2048; float32 for the fp32 rows,
float32-accumulated bf16 products for the bf16 rows):

| inputs / config | max abs | mean abs |
| --- | --- | --- |
| fp32 inputs, default (`HiFi4`, fp32 dest) | 4.8e-3 | 8.4e-4 |
| fp32 inputs, `HiFi3` + `fp32_dest_acc_en` | 5.3e-3 | 9.2e-4 |
| fp32 inputs, `HiFi2` + `fp32_dest_acc_en` | 2.9e-2 | 5.2e-3 |
| fp32 inputs, `LoFi` | 1.2e-1 | 1.7e-2 |
| bf16 inputs, default | 6.8e-2 | 6.5e-3 |
| bf16 inputs, `HiFi4` + `fp32_dest_acc_en` | 8.3e-3 | 1.1e-3 |

The best available matmul is therefore about 1e-3 relative. A CPU fp32
reference that dequantizes bf16 weights and computes in fp32 reaches ~1e-4, so
device hidden states stay a few times further from the oracle than an HF bf16
run does, and the gap widens with context length (the model amplifies
per-layer noise through 24 residual branches). `ttnn.transformer.
scaled_dot_product_attention` accepts bf16-family inputs only. In fp32 mode the
prefill therefore runs attention with explicit float32 matmuls, a float32
softmax, and a host-side causal/window mask (`_attention_exact`, 256-row query
blocks), while decode defaults to the bf16 SDPA kernel over a bf16 KV cache.
`--decode-attention fp32` swaps that for a float32 cache and a float32
`_attention_exact_decode` (per-step additive mask over the used rows).

## Limits

- Decode attention defaults to the bf16 SDPA kernel, so the KV cache carried
  across tokens is bf16 even in fp32 mode. `--decode-attention fp32` removes
  that rounding at ~4.5x the per-token cost; at 430+32 tokens it produced the
  same 32 tokens as the default, so the default stays on the fast path.
- The bf16 mode is the throughput mode; it is not an accuracy gate. It matched
  31/32 tokens on the short prompt and 1/8 on the 430-token prompt.
- The runner is single-card (`1x1` mesh on device 0) and greedy; there is no
  sampler, batch scheduler, or service wrapper.
- The checkpoint is not pinned by the derivation; `--checkpoint-dir` selects a
  local Hugging Face download. Add a fixed-output derivation before any
  reproducibility claim.
