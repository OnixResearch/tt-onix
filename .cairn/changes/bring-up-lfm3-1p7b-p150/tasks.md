# Tasks: Bring up LFM3-1.7B on one P150

## Phase 1: Feasibility

- [x] [serial] F1 Read the checkpoint's `config.json`, `modeling_lfm3.py`, and `lfm3_conv.py`; record the layer mix, the convolution window, the partial-RoPE split, the per-head gate, and the sandwich norms. Confirm the release blog's architecture claims against the checkpoint files. r[tenstorrent.lfm3.p150.gpu]
- [x] [serial] F2 Measure the packaged runtime's matmul fidelity across dtype and `math_fidelity` configurations, and confirm the SDPA kernels accept bf16 only. r[tenstorrent.lfm3.precision.disclosure.matmul]

## Phase 2: Implementation

- [x] [serial] I1 Add `lfm3_runner.model`: checkpoint loading with transposed matmul weights, the fused QKV and gate/up projections, the partial/full RoPE tables, the row-major convolution window, and the decode steps. r[tenstorrent.lfm3.p150.gpu]
- [x] [serial] I2 Add the float32 mode: wide residual stream, explicit float32 attention with a host-built additive mask in 256-row blocks, and the bf16 SDPA exception for decode. r[tenstorrent.lfm3.accuracy.oracle]
- [x] [serial] I3 Add `--decode-attention` with the bf16 SDPA default and the float32 cache path. r[tenstorrent.lfm3.decode.attention]
- [x] [serial] I4 Add `lfm3-run` and `lfm3-check`, the prompt module, the writable `TT_METAL_HOME` launcher, the package derivation, and the `.#lfm3` export. r[tenstorrent.lfm3.p150.gpu.checkpoint]
- [x] [serial] I5 Add the CPU fp32 oracle script outside the derivation with the transformers-version rationale. r[tenstorrent.lfm3.accuracy.oracle]
- [x] [serial] I6 Add `docs/lfm3-1.7b-p150.md` with the architecture map, measurement tables, precision probe, and non-claims. r[tenstorrent.lfm3.precision.disclosure]

## Phase 3: Verification

- [x] [serial] V1 Build `.#lfm3` with its install check and run `lfm3-check` against the 17-token oracle: 32/32 greedy tokens, prefill logits PCC 0.99927, final-norm hidden PCC 0.9982, layers 0-12 at PCC >= 0.9998. r[tenstorrent.lfm3.accuracy.oracle.short]
- [x] [serial] V2 Run the 430-token prompt for 32 generated tokens: 32/32 with the default decode attention at 26.1 tok/s and 32/32 with `--decode-attention fp32` at 5.8 tok/s; prefill 918 tok/s warm. r[tenstorrent.lfm3.accuracy.oracle.long] r[tenstorrent.lfm3.decode.attention.default] r[tenstorrent.lfm3.decode.attention.wide]
- [x] [serial] V3 Run the bf16 mode on both prompts: 31/32 on the short prompt (first divergence at index 18) and 0/32 on the 430-token prompt. Recorded as a throughput mode, not a gate. r[tenstorrent.lfm3.precision.disclosure.bfloat16]
- [x] [serial] V4 Run `nix build .#lfm3` (install check imports TT-NN), `ruff check`/`ruff format --check` on the runner and the oracle, and the packaged entry points on hardware. r[tenstorrent.lfm3.p150.gpu.generate]

## Phase 4: Completion

- [x] [serial] D1 Record the receipts and every non-claim in `evidence/`. Do not pin the checkpoint, do not add a service, do not add tracing or batching. r[tenstorrent.lfm3.precision.disclosure]
