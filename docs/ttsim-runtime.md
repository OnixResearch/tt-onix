# TTsim runtime bundle

The `ttsim` package provides complete simulator runtime directories. Each supported directory contains these files:

- the exact host-specific release asset;
- a canonical `libttsim.so` link;
- the matching `soc_descriptor.yaml` from TT-Metal 0.74.0.

The package supports `x86_64-linux` and `aarch64-linux`. It never uses an x86-64 asset on an AArch64 host.

## Supported single-chip runtimes

| Wrapper | Chip | Topology | Dispatch |
|---|---|---|---|
| `ttsim-blackhole` | Blackhole | `x1` | slow |
| `ttsim-wormhole` | Wormhole | `x1` | slow |
| `ttsim-blackhole-sfpu` | Blackhole | `x1` | slow |
| `ttsim-wormhole-sfpu` | Wormhole | `x1` | slow |

The normal wrappers set `TT_METAL_SIMULATOR` and `TT_METAL_SLOW_DISPATCH_MODE=1`. They also remove an inherited `TT_MESH_GRAPH_DESC_PATH` because it can select an unrelated physical mesh.

Use an `-sfpu` wrapper only when the workload can use `SFPLOADMACRO`. These wrappers also set `TT_METAL_DISABLE_SFPLOADMACRO=1`. TTsim 1.9.2 does not support that macro.

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

## Multi-chip boundary

The release includes `bh_x2`, `bh_x32`, `wh_x2`, `wh_x8`, and `wh_x32` assets. The package keeps these exact host assets in its asset directory. It does not expose them as runnable simulator environments.

TTsim 1.9.2 requires the unmerged `mcraighead/mc-p300` TT-Metal integration for these assets. TT-Metal 0.74.0 has mock-cluster plumbing, but its package has no reviewed `blackhole_P300` cluster descriptor.

The `ttsim-blackhole-x2` command therefore stops before simulation. It prints the blocker receipt and exits nonzero. The package stores the same receipt at `share/ttsim/blackhole-x2-blocker.json`.

Do not use a simulated P300 topology as P150x2 evidence.

## Claim boundary

A passing simulator receipt proves only the observed simulated software behavior. It does not prove these properties:

- physical placement;
- hardware profiling;
- performance;
- thermals;
- device health;
- silicon success.

Use silicon for those claims after the bounded simulator smoke passes.
