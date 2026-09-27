# p5: snapshot memory and the simulator blocker (2026-09-27)

Snapshot memory was measured on hardware. The simulator restore did not run, because
the TTsim blocker that stopped s5 also covers this path. No simulator receipt is claimed.

## Snapshot memory

Each root reports the `device-dram` event that the service emits after startup warm-up.
Both roots ran package `/nix/store/43ayflq56m4yhx3p5n6ds6mishwki1gd-qwen38-0.1.0` on
britton-desktop (P150x2, PCIe devices 0 and 1). The package is built from commit
`7a15fdd`.

| Root | Allocated bytes | Free bytes | Largest free per bank |
| --- | --- | --- | --- |
| Cold (`--continuous-slot-policy` only) | 23,185,597,440 | 9,919,391,744 | 1,104,700,416 |
| Cached (adds `--prefix-cache-policy`) | 23,550,501,888 | 9,554,487,296 | 1,104,700,416 |

- The cache costs 364,904,448 B, about 348 MiB. The admission gate allows up to
  536,870,912 B.
- Policy `qwen38-p150x2-prefix-cache-v1` holds `state_slots = 4`. Each state is therefore
  91,226,112 B, about 87 MiB. A state is one carried-GDN buffer: recurrent state plus
  convolution carry. The design estimated 72 MiB of raw recurrent state per chip, before
  convolution.
- Cached KV takes no extra DRAM. Entries pin blocks of the existing pool, up to 1,024
  blocks under the policy's `maximum_cached_blocks`.
- The largest free region per bank does not change, so the state buffers do not
  fragment the pool that the decode trace needs.

## Simulator restore

The cache runs only inside the continuous-slot scheduler, and that scheduler is TP=2
only. A simulated restore would need `libttsim_bh_x2.so`. AGENTS.md admits the two-chip
model only after the single-chip GDN smoke passes, and that smoke still fails:

```text
ttsim-gdn-fp32-reference-smoke> [7470] ERROR: UnsupportedFunctionality: tensix_matmul_op: ALU_ACC_CTRL_Zero_Flag_disabled_src=1 (keep SrcB denormals) not modeled
```

The run used TTsim 1.10.6 and TT-Metal 0.77.0 on 2026-09-27. The command, inputs and
derivation are recorded in
`../../qwen38-continuous-slot-scheduler/evidence/s5-simulator-blocker-and-physical-screen.md`.
The cached-state path adds no operation that could avoid the GDN kernels, so the same
blocker applies. The restore was therefore tested on hardware; see
`p6-hardware-screen.md`.
