# TTsim runtime bundle

The `ttsim` package builds pinned upstream release 1.10.6 source at
`a5276e15a52729465c8ddd71f0965df4ca2f0cc1` with the repository's reviewed
Blackhole PACR zero-flag model patch. It provides complete single-chip simulator
runtime directories. Each supported directory contains these files:

- the host-native library built from that patched source;
- a canonical `libttsim.so` link;
- the matching `soc_descriptor.yaml` from the selected TT-Metal package.

The package supports `x86_64-linux` and `aarch64-linux`. Every selected library is
built for its host, including the retained multi-chip assets. The runtime manifest
records the source revision/hash, local model revision and patch SHA-256.
`upstreamReleaseHash` values are explicitly the unmodified upstream binary
identities for audit/negative-host fixtures, not hashes of the source-built output.
Numerical receipts continue to bind the actual installed library bytes.

## Supported single-chip runtimes

| Wrapper | Chip | Topology | Dispatch |
|---|---|---|---|
| `ttsim-blackhole` | Blackhole | `x1` | slow |
| `ttsim-wormhole` | Wormhole | `x1` | slow |
| `ttsim-blackhole-sfpu` | Blackhole | `x1` | slow |
| `ttsim-wormhole-sfpu` | Wormhole | `x1` | slow |

The normal wrappers set `TT_METAL_SIMULATOR` and `TT_METAL_SLOW_DISPATCH_MODE=1`. They also remove an inherited `TT_MESH_GRAPH_DESC_PATH` because it can select an unrelated physical mesh.

Use an `-sfpu` wrapper only when the workload can use `SFPLOADMACRO`. These wrappers also set `TT_METAL_DISABLE_SFPLOADMACRO=1`. TTsim 1.10.6 does not support that macro.

The wrappers run the command that follows the wrapper name. For example:

```console
nix shell .#ttsim --command ttsim-blackhole <program> [arguments...]
```

## Blackhole smoke

Run the bounded Blackhole smoke with this command:

```console
nix run .#ttsim-blackhole-smoke -- --receipt /tmp/ttsim-blackhole-receipt.json
```

The smoke has a 3,600-second timeout. It uses packaged TT-Metal and TT-NN. It opens one simulated device, completes TT-Metal firmware initialization, runs `ttnn.ones` and `ttnn.add`, synchronizes the device, and closes it.

The runner always classifies these failure types:

- timeout;
- unsupported operation;
- nonzero exit;
- incomplete device open;
- incomplete firmware compilation;
- incomplete representative operations;
- incomplete close.

The receipt binds the library, descriptor, host system, TT-Metal identity, dispatch mode, workload, timeout, and outcome. File and package identities use BLAKE3.

## Blackhole PACR zero-flag model

The local `blackhole-pacr-zero-flags-v1` model revision admits
`Disable_pack_zero_flags` on Blackhole's already supported **dense-L1** PACR
path. This is output sideband metadata, not `Dst` row validity. The package-matched
Blackhole LLK `cpack_common.h::reconfigure_packer_l1_acc` explains why an all-zero
face must not set output zero flags while accumulating into unknown existing L1.
The [pinned ISA TDMA-RISC contract](https://github.com/tenstorrent/tt-isa-documentation/blob/02d919f4c9c85ee23ce251b0e95c52300ec617d8/WormholeB0/TensixTile/TDMA-RISC.md)
identifies `AllZeroFlags` as packer sideband metadata; its
[Blackhole packer chapter](https://github.com/tenstorrent/tt-isa-documentation/blob/02d919f4c9c85ee23ce251b0e95c52300ec617d8/BlackholeA0/TensixTile/TensixCoprocessor/Packers/README.md)
explicitly shares that description with Wormhole.

The simulator writes actual numerical zero bytes to dense SRAM, and its existing
L1 accumulation reads and adds those actual bytes. The zero-flag sideband has no
consumer on this supported path: compression, metadata extraction and metadata
FIFO output remain rejected. Both values of the output flag therefore preserve
the same observable data, without bypassing `ZEROACC`/`Dst` valid-row masking or
changing any format conversion, pack rounding, or L1 arithmetic. Wormhole retains
its original unsupported guard. This does not implement compressed tiles or
zero-flag metadata reads, nor the separate GDN matmul control flag.

The normal package check executes the actual patched PACR implementation on zero,
stale invalidated Dst, mixed valid/invalid rows and signed data, with and without
L1 accumulation and both flag values. It also checks BF16 tie narrowing and the
overwrite→zero-accumulate→overwrite transition, preserves `Dst` data/validity, and
requires the unrelated compression/threshold/metadata guards to remain failures.

The unchanged installed Qwen rounded-register checker now completes exact BF16
numerical comparison at B1 and B8 (K=5120, local N=8704) through packaged TT-Metal
0.77.0: both baseline and candidate report zero unequal elements, zero maximum
absolute error and zero relative L2 error, followed by clean device close.
The normal source-built package and runtime-bundle/ABI check also pass.
The [rounded-register receipt](../pkgs/qwen36/evidence/2026-10-23/rounded-register-swiglu.json)
binds the pinned source, patch, model revision, exact commands and library bytes;
it retains the original unsupported result separately. This is single-chip
component qualification only, not full-model, multi-chip, physical or performance
acceptance. Native metalium-rs still uses its distinct pinned TTsim 1.10.9.

## Multi-chip boundary

The release includes `bh_x2`, `bh_x32`, `wh_x2`, `wh_x8`, and `wh_x32` configurations.
The package builds these host-native libraries from the same patched source and
keeps them in its asset directory. It does not expose them as runnable simulator
environments.

TTsim 1.10.6 documents incomplete multi-chip UMD support beyond `wh_x2`. Its README names `micah/native-sim` as the validated TT-Metal branch for Blackhole multi-chip simulation. The package has no admitted multi-chip runtime or reviewed `blackhole_P300` cluster descriptor.

Release 1.10.6 also publishes `bh_x4` assets. This package does not yet include or admit those assets.

The `ttsim-blackhole-x2` command therefore stops before simulation. It prints the blocker receipt and exits nonzero. The package stores the same receipt at `share/ttsim/blackhole-x2-blocker.json`.

Do not use a simulated P300 topology as P150x2 evidence.

## GDN numerical check

The [0.77 port](tt-metal-0.77-port.md) adds `checks.x86_64-linux.ttsim-gdn-reference-smoke`. This check compares the fused recurrence with a NumPy reference and checks state carry and invalid inputs.

TTsim 1.10.6 currently stops at an unsupported FP32 matmul control flag. The check remains failing. The basic `ttnn.ones`/`ttnn.add` smoke does not replace this numerical gate.

## Claim boundary

A passing simulator receipt proves only the observed simulated software behavior. It does not prove these properties:

- physical placement;
- hardware profiling;
- performance;
- thermals;
- device health;
- silicon success.

Use silicon for those claims after the bounded simulator smoke passes.
