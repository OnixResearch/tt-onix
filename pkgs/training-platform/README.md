# Dual-P150A TT-XLA training platform

One immutable TT-XLA training runtime for two Blackhole P150A cards cabled as
one `p150_x2` (1x2) mesh.

This package establishes platform readiness only. It does not establish
training correctness, gradient parity, convergence, speedup, or thermal
endurance.

## Runtime identity

The compatibility manifest lives in `manifest.ncl` and exports to JSON. Every
component carries an immutable identity. Identities were resolved from the
listed upstream authorities on 2026-08-04.

| Component | Identity |
|---|---|
| Runtime image | `ghcr.io/tenstorrent/tt-xla/tt-xla-base-ubuntu-24-04@sha256:8dcad76548126bb33c13ce554192cb9f27612fa3766b7c7e7ae6f591f2ab6503` |
| PJRT plugin | `pjrt-plugin-tt 1.5.0.dev20260804002041`, wheel sha256 `ae57e7a2e4f7c1c9533135e14db5a33201caa3ff1aa2278de9a88ebc9d4d3fe1` |
| TT-XLA source | `tenstorrent/tt-xla@10b111f3885a598958b0c4e393a07d672584736a` |
| TT-Blacksmith source | `tenstorrent/tt-blacksmith@6e44259c7f4342709ae7977cc3715cf5e2e19dac` |
| PyTorch | `torch 2.11.0+cpu` (cp312), wheel sha256 `f82e2ae20c1545bb03997d1cc3143d94e14b800038669ee1aca45808a9acc338` |
| PyTorch/XLA | `torch-xla 2.9.0+gitb230007` (cp312), wheel sha256 `4318d8cce4f9de38d497712b9e80cb74746dc16dc8e2a15e555c417711757481` |
| JAX | `jax 0.7.1`, `jaxlib 0.7.1` |
| Driver | TT-KMD `2.10.1-pre` |
| Firmware floor | bundle `18.8.0.0` |

The pinned TT-Blacksmith revision pins `pjrt-plugin-tt==1.5.0.dev20260804002041`
in `env/xla_requirements.txt`. The bundle is internally coherent.

The recorded host baseline is `baseline-2026-08-04.json`.

## Hardware prerequisites

- Two Blackhole P150A cards connected as one 1x2 mesh (QSFP-DD fabric cables).
- The `p150_x2` mesh graph descriptor from the packaged TT-Metal. The NixOS
  module exports it through `TT_MESH_GRAPH_DESC_PATH` when
  `hardware.tenstorrent.meshName = "p150_x2"`.
- Reserved 1G hugepages mounted at `/dev/hugepages-1G`. The validated host
  reserves eight pages.
- TT-KMD driver and firmware at or above the manifest floor on both cards.
- Docker for the runtime container.

`nixos/examples/dual-p150a-training.nix` declares all of these.

## Commands

Validate the device-free platform layer:

```sh
nix build .#checks.x86_64-linux.training-platform
```

Run the bounded hardware smoke (preflight, mesh open, one collective-free
tensor operation, mesh close, restoration, health capture):

```sh
TOOLS=$(nix build .#packages.x86_64-linux.training-platform-tools --no-link --print-out-paths)
"$TOOLS/bin/training-platform-smoke" \
  --manifest "$TOOLS/share/training-platform/manifest.json" \
  --config "$TOOLS/share/training-platform/platform.json" \
  --descriptor "$TOOLS/share/training-platform/p150_x2_mesh_graph_descriptor.textproto" \
  --evidence-root /var/lib/tenstorrent-training/evidence
```

The smoke writes a BLAKE3-bound receipt to the evidence root and validates it
with `training-platform-core validate-receipt`.

The launcher rejects ambient `PYTHONPATH`, `PYTHONHOME`, `TT_METAL_HOME`, and
`TT_METAL_RUNTIME_ROOT`. These variables can silently replace the admitted
runtime.

## Recovery

If the smoke fails before `mesh_open`, no device state changed. Read the
admission diagnostic.

If the smoke fails inside the container, the container is removed by
`--rm`. Check that no process owns `/dev/tenstorrent/0` or
`/dev/tenstorrent/1` before you retry (`fuser /dev/tenstorrent/0`). The
orchestrator runs the same check in its `restore` stage.

If firmware or driver admission fails, do not flash firmware from this
package. Record the blocker and follow the Tenstorrent firmware update
procedure outside this runtime.

## Evidence and non-claims

Receipts bind the bundle, configuration, and launch-plan identities with
BLAKE3. Receipts exclude credentials, raw logs, serial numbers, and absolute
private paths.

A passing receipt proves:

- The admitted runtime identity started on this host.
- The dual-card preflight passed with the observed facts.
- The `p150_x2` mesh opened, ran one bounded collective-free tensor
  operation, and closed.
- Device ownership was restored.

A passing receipt does not prove training correctness, gradient parity,
convergence, speedup, or thermal endurance. Model work needs a separate
Cairn change.

## Authority sources

- <https://docs.tenstorrent.com/tt-xla/getting_started.html>
- <https://docs.tenstorrent.com/tt-blacksmith/src/getting-started.html>
- `tenstorrent/tt-xla` at `10b111f3885a598958b0c4e393a07d672584736a`
- `tenstorrent/tt-blacksmith` at `6e44259c7f4342709ae7977cc3715cf5e2e19dac`
- Packaged `p150_x2_mesh_graph_descriptor.textproto` (BLAKE3
  `83b6c1ff57a7a4686cd38b1a29ac38540e4c86bfd8e01efa989a2f10ee1b970b`)
