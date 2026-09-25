## Why

The B=1 P150x2 decode step is 54.3 ms on the served composite GDN route. The
earlier attribution charged the whole 0.549 ms GDN mixer to the recurrence loop.
Per-op prefix timing on silicon disagrees with that. The fused recurrence kernel
takes 27 µs of the 459 µs fused GDN layer. The composite route spends about
300 µs per layer on glue between the in and out projections. That glue
includes the conv shift and multiply-accumulate chain, slices, reshapes,
`repeat_interleave`, permutes, and typecasts. Across 48 layers this is about
10 to 14 ms of every step. The projections run near the same effective DRAM
bandwidth as the MLP. The collectives are already as fast as the alternatives
measured on this mesh. See `evidence/attribution.md`.

On 2026-09-24 the owner approved a numerical quality gate for the fused GDN route
instead of bit-exact greedy parity with the composite route. The fused route is
more precise than the composite route. Bit-exact parity blocked every
GDN fusion from the greedy serving path.

## What Changes

- Add decode fusion mode `step`. Two custom programs replace the decode chain
  between the `[qkv|z|a|b]` projection and the out projection. Each GDN layer
  runs those programs through `ttnn.generic_op`.
  - `conv_prep` performs the depthwise causal conv and SiLU for the q/k/v
    channels, SiLU for the gate, and an fp32 copy of the a/b logits. It writes
    one persistent fp32 buffer and shifts the conv state in place.
  - `recurrence` handles each `(batch, value head)`. It performs the q/k L2
    norms, beta and decay, gated delta rule, output read, and gated RMSNorm.
    It writes the out projection's bf16 input in place of the composite output
    and updates the fp32 recurrent state in place. The carried state is never
    unpacked through the 19-bit source registers.
- Route greedy requests to `step` when the GDN layers start in `step`, and add
  that route to the router core. Sampled requests keep the sampled-fusion route.
- Add a NixOS option `hardware.tenstorrent.qwen38.gdnDecodeStep` that defaults to
  on and sets `QWEN35_GDN_DECODE_FUSION=step`. It rejects the combination with MTP
  speculation, which has no evidence on this route.

## Impact

- **Files**: `pkgs/qwen36/gdn-decode-step.patch` (new `decode_step.py`, six
  kernels, the `tp.py` branch, and the mode policy),
  `pkgs/qwen36/default.nix`, `pkgs/qwen36/check.py`,
  `pkgs/qwen36/service/qwen36_service/{sampled_mode_router_core,batch_service}.py`,
  `nixos/module.nix`, and `flake-modules/onix-packages.nix`.
- **Testing**: the qwen38 package build and device-free check. The router
  and geometry probes cover positive and negative cases. The NixOS service check
  covers the defaults, opt-out, and MTP rejection. The hardware kernel test
  compares against a float64 reference. The hardware model evaluation covers
  step time, teacher-forced agreement, and passage NLL.
- **Production**: the host configuration lives outside this repository. It
  changes only when that host adopts this package and module.
