## Why

The optimized runtime still dispatches many generic TT-NN operations for each RWKV-7 layer and token. Recent work removed individual casts, reshapes, transposes, and pointwise operations. Those gains are real but too small to reach thousands of tokens per second.

The recurrent WKV path has a stable equation and fixed 64-wide heads. A Blackhole-specific fused operator can update FP32 state and produce BF16 attention output without separate reshape, transpose, cast, and pointwise programs.

## What Changes

- Add a package-matched RWKV-7 WKV operator for 64-wide Blackhole heads.
- Fuse the FP32 recurrent update, state readout, attention correction, and BF16 output boundary.
- Keep the functional equation and reference implementation separate from device orchestration.
- Add positive and negative tensor, state, shape, precision, and synchronization tests.
- Run simulator smoke and all applicable LLK hazard and performance audits before silicon.
- Select the fused path only when exact parity and stable physical improvement pass.

## Impact

- **Runtime**: `pkgs/rwkv7-p150x2` gains a specialized WKV execution path and safe fallback.
- **Kernel**: new TT-Metal or LLK source enters the runtime closure.
- **Evidence**: profiler and physical receipts bind launch reduction to measured throughput.
- **Testing**: reference-equation, malformed-shape, precision, and synchronization cases are required.

## Out of Scope

- Reusing the packaged gated-delta operator, which requires incompatible 128-wide chunks and dimensions.
- Changing the checkpoint, tokenizer, sampling, or FP32 recurrent-state contract.
- Claiming speed from launch count alone.
- Replacing all model matrix products with custom kernels.
