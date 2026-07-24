# tenstorrent.nix

A Nix Flake for using Tenstorrent hardware with Nix/NixOS. Packages are staged
here in nixpkgs form before being upstreamed, so the overlay and module are
drop-in replacements for what nixpkgs provides.

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

- [RossComputerGuy/tenstorrent.nix](https://github.com/RossComputerGuy/tenstorrent.nix) — upstream package history used as this repository's base.
- [onixcomputer/onix-core](https://github.com/onixcomputer/onix-core) — source repository for the migrated Onix package snapshot and its fine-grained development history.
