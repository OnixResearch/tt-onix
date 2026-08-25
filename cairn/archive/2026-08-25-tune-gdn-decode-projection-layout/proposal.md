## Why

The measured Qwen3.8-27B P150x2 decode step attributes roughly half of device
time to the gated-delta mixers. Their TP projections run on ttnn AUTO matmul
layouts with a compute-kernel config but no program config, which the
reviewed Qwen3.6-A3B bundle measured as the worst-tuned decode shape class:
wide-1D multicast and DRAM-sharded layouts reached `1.71x` to `2.7x` per op
at batch one on comparable projection shapes.

## What Changes

- Add an operator-enabled decode projection-layout candidate for the GDN
  mixer: explicit wide-1D multicast or DRAM-sharded program configs behind a
  default-off environment gate.
- Keep the current AUTO layouts as the untouched production path.
- Evaluate the candidate with matched eager-versus-candidate evidence under
  the existing decode admission requirements before any claim.

## Impact

- **Files**: adopted-tree GDN TP projection call sites (new patch), layout
  selection core, device-free check probes.
- **Testing**: positive and negative device-free tests, Python compilation,
  Ruff, packaged check, Cairn validation, and one matched physical A/B pair.
