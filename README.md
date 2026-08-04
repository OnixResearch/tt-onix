# tenstorrent.nix

A standalone private Nix flake for using Tenstorrent hardware with Nix/NixOS.
The `OnixResearch/tenstorrent.nix` `main` branch is the sole repository
authority; no external upstream remote is configured.

## Overlay

```nix
nixpkgs.overlays = [ tenstorrent.overlays.default ];
```

Provides top-level `tt-umd`, `tt-logger`, `tt-metal`, `tt-smi`, `tt-burnin`,
`tt-topology`, `tt-system-tools`, and `luwen`, extends every Python package set
with `pyluwen`, `tt-flash`, and `tt-tools-common`, and extends kernel package
sets with `tt-kmd`.

## NixOS module

```nix
{
  imports = [ tenstorrent.nixosModules.default ];
  hardware.tenstorrent.enable = true;
}
```

Matches the upstream `hardware.tenstorrent` module, replaces it when both are
present, and applies the overlay automatically.

## Fast local iteration

Package-only changes do not need a NixOS activation. Run the exact package from
the current checkout:

```sh
nix run .#ttwkv7 -- shape-test
nix run .#ttwkv7-constant-probe -- self-test
nix run .#ttwkv7-data-movement -- self-test
```

Each command rebuilds only the affected derivations. Nix reuses the unchanged
host binaries, kernels, and dependencies from the store.

Build a stable result link when you need to inspect or run several commands:

```sh
nix build .#ttwkv7 -o result-ttwkv7
./result-ttwkv7/bin/wkv7 shape-test
./result-ttwkv7/bin/wkv7-constant-probe self-test
```

Run the focused device-free checks before the final host integration check:

```sh
nix build .#checks.x86_64-linux.development-interface
nix build .#checks.x86_64-linux.ttwkv7-architectures
```

A NixOS activation is still necessary for kernel modules, udev rules, system
environment changes, and service changes. Use the full host closure as the
final integration gate, not as the package inner loop.

## Onix packages

On `x86_64-linux`, the flake also publishes:

- `rwkv-lab` and `rwkv-layer-harness` for deterministic device-free RWKV-7 validation;
- `ttwkv7` plus bounded and persistent operator wrappers;
- retained partial and passing ttWKV7 evidence validators;
- `ttwkv7-owner-control`, `tt-vibethinker-bench`, and `ttsim`.

The focused checks include positive and negative fixtures and do not acquire a
Tenstorrent device. Building `ttwkv7` requires allowing its unfree package license.
See [`docs/migration-provenance.md`](docs/migration-provenance.md) for the exact
source boundary and non-claims.

## References

### Project and model references

- [onixcomputer/onix-core](https://github.com/onixcomputer/onix-core) — source repository for the migrated Onix package snapshot and its development history.
- [BlinkDL/RWKV-LM](https://github.com/BlinkDL/RWKV-LM) — upstream RWKV training and inference source used as an implementation reference.
- [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) — parent backend reference for RWKV-7, CUDA, Vulkan, and ggml operator work.
- [marty1885/llama.cpp](https://github.com/marty1885/llama.cpp) — package upstream for the Metalium backend and experimental RWKV trees.

### Tenstorrent platform references

- [tenstorrent/tenstorrent.github.io](https://github.com/tenstorrent/tenstorrent.github.io) — source for the public Tenstorrent documentation site.
- [tenstorrent/tt-metal](https://github.com/tenstorrent/tt-metal) — package-matched TT-Metalium, TT-NN, tests, and technical documentation source.
- [tenstorrent/tt-isa-documentation](https://github.com/tenstorrent/tt-isa-documentation) — instruction-set and hardware behavior reference.
- [tenstorrent/tt-low-level-documentation](https://github.com/tenstorrent/tt-low-level-documentation) — LLK, NoC, overlay, and kernel performance reference.

### Tenstorrent host and simulation references

- [tenstorrent/tt-umd](https://github.com/tenstorrent/tt-umd) — package-matched user-mode driver and host interface source.
- [tenstorrent/tt-kmd](https://github.com/tenstorrent/tt-kmd) — package-matched kernel driver and device-node source.
- [tenstorrent/luwen](https://github.com/tenstorrent/luwen) — package-matched chip discovery, diagnostics, and firmware communication source.
- [tenstorrent/ttsim](https://github.com/tenstorrent/ttsim) — device-free simulator source and ABI reference.

### Tenstorrent operations references

- [tenstorrent/tt-tools-common](https://github.com/tenstorrent/tt-tools-common) — shared behavior and schemas for Tenstorrent operations tools.
- [tenstorrent/tt-smi](https://github.com/tenstorrent/tt-smi) and [tenstorrent/tt-topology](https://github.com/tenstorrent/tt-topology) — diagnostics and topology sources.
- [tenstorrent/tt-flash](https://github.com/tenstorrent/tt-flash) — firmware update and compatibility-check source.
- [tenstorrent/tt-burnin](https://github.com/tenstorrent/tt-burnin) — hardware stress and validation source.
- [tenstorrent/tt-system-tools](https://github.com/tenstorrent/tt-system-tools) — incident collection and `tt-oops` source.
