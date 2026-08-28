# Runtime-uplift disposition

Date: 2026-08-20

## Verdict

`UNSELECTED_UNADMITTED`

No runtime-uplift candidate was selected.
TT-Metal `0.74.0-rc6` remains the Qwen3.8 production runtime.

## Current closure

- Runtime tag: `v0.74.0-rc6`
- Package-matched runtime source: `c49bb7625e6a495337483353a5228339fece2c11`
- Overlaid Qwen3.8 model source: `d102e7f4512060e09e580e40cf26a870bfa88c55`
- Production mesh: `P150x2`
- Production service state at disposition: `active`

The package retains these material compatibility paths:

- `gdn-seq-fallback.patch` selects sequential GDN when the fused operation is absent.
- `tp2-runtime-compat.patch` avoids the TP4-tuned fused prefill reduction on two devices.
- The same patch uses separate MLP matrix products when the installed binding lacks `fuse_swiglu`.

## Missing candidate

No isolated candidate binds one reviewed runtime revision to a package-matched model source, patch set, compiler, Python ABI, firmware, driver, and topology.

No candidate closure, capability receipt, simulator receipt, physical service receipt, or admission receipt exists.

## Decision

The current runtime and all compatibility fallbacks remain selected.
No flake input changed, and `flake.lock` was not modified.

A future evaluation must use a new validated Cairn change and one immutable candidate closure.

## Non-claims

- This disposition does not show that a newer runtime is incompatible.
- This disposition does not compare runtime performance.
- Public version or feature names do not prove Qwen3.8 behavior.
- No simulator or physical candidate run was performed.
