## Why

The user asked to get `LiquidAI/LFM3-1.7B` working and optimized on the
Tenstorrent cards. LFM3-1.7B is a tri-hybrid decoder: 24 layers, of which 13 are
gated short convolutions (kernel 7), 5 are sliding-window attention (window 256,
full RoPE, theta 1e4), and 6 are global attention (partial RoPE 0.5, theta 1e6).
Hidden 2048, FFN 7680, 16 query / 4 KV heads, head_dim 128, per-head sigmoid
output gate, sandwich RMSNorm, tied 128K embedding and LM head.

No packaged path runs it. The model is custom-code `Lfm3ForCausalLM` with no
`tt_transformers` equivalent: the convolution stack, the per-head gate, and the
partial-RoPE split have no generic HF walker. The repository's device stack is
TT-Metal 0.77 with TT-NN, so the bring-up is repo-authored TT-NN code.

The measured matmul fidelity of the packaged runtime is the binding constraint:
every `ttnn.linear` variant lands near 1e-3 relative on float32 inputs, so the
useful acceptance gate is agreement with a CPU fp32 oracle on the greedy stream
plus a reported per-layer deviation, not bit equality.

## What Changes

- Add `pkgs/lfm3` with a repo-authored runner (`lfm3_runner.model`) that
  implements the checkpoint's math: fused QKV projection, fused gate/up FFN,
  per-head attention gate, partial/full RoPE, sandwich norms, and a width-7
  causal depthwise convolution with a six-row recurrent window for decode.
- Add two precision modes. `--dtype float32` keeps the residual stream wide and
  runs prefill attention with explicit float32 matmuls, a float32 softmax, and a
  host-built causal/window mask; `--dtype bfloat16` is the throughput mode.
- Add `--decode-attention {sdpa,fp32}`. The default keeps the bf16 SDPA decode
  kernel over a bf16 KV cache; `fp32` widens the cache and runs a float32
  attention over the used rows.
- Add `lfm3-run` (device generation with timings) and `lfm3-check` (per-layer
  hidden-state, prefill-logit, and decode-token comparison against an oracle
  dump), exported as `.#lfm3`.
- Add `pkgs/lfm3/oracle/reference.py`, the CPU fp32 oracle. It stays outside the
  derivation because the checkpoint's remote code needs `transformers >= 5.12.1`
  and the nixpkgs transformers does not export the API it imports.
- Add `docs/lfm3-1.7b-p150.md` with the architecture map, the measurement
  tables, the matmul-fidelity probe, and the non-claims.

## Impact

- **Files**: new `pkgs/lfm3/`, an export in
  `flake-modules/onix-packages.nix`, `docs/lfm3-1.7b-p150.md`, and this change.
- **Testing**: the package install check (the runner imports against the packaged
  TT-NN), `lfm3-check` against the fp32 oracle for the 17-token and 430-token
  prompts, greedy agreement for 32 generated tokens in both modes, and the
  decode-attention variants.
- **Not performed**: no checkpoint pin, no service, no mesh beyond device 0, no
  sampler or stop policy, no traced decode, no batch serving. No hardware state
  was changed beyond running the device.
