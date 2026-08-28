## Why

The current `ttsim` derivation installs simulator libraries but not the adjacent `soc_descriptor.yaml` that TT-Metal requires. It also fetches unsuffixed release assets and exposes only `x86_64-linux`, while upstream publishes separate `_aarch64` assets.

The flake check proves that the files install. It does not prove that TT-Metal can open a simulated Blackhole device. This leaves the simulator-before-hardware workflow without a usable package boundary.

## What Changes

- Package architecture-specific simulator assets with exact hashes.
- Build one runtime directory per chip and topology with the required `soc_descriptor.yaml` beside the selected library.
- Add wrappers that set `TT_METAL_SIMULATOR`, slow dispatch, and supported simulator workarounds.
- Add a bounded Blackhole `1x1` TT-Metal smoke before any multi-chip simulator work.
- Add explicit simulated receipts and non-claims.
- Gate `bh_x2` behind package-matched cluster-descriptor and TT-Metal compatibility evidence.

## Impact

- **Packages**: `pkgs/ttsim`, `pkgs/tt-metal`, flake package exports, and simulator checks.
- **Interfaces**: named Blackhole and Wormhole runtime wrappers become the supported entry points.
- **Testing**: asset-layout checks, ABI checks, negative fixtures, and one device-free simulator smoke.
- **Documentation**: simulator selection, timeouts, supported systems, and evidence limits.

## Out of Scope

- Physical placement, hardware profiling, performance, or silicon acceptance claims.
- Quasar host-MMIO support that upstream does not provide.
- Multi-chip support on an unmatched TT-Metal branch.
