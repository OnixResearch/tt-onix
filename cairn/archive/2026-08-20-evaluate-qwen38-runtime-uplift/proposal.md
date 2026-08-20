## Why

A Qwen3.8 runtime uplift needs one immutable candidate runtime and a package-matched model source.
No such candidate was selected or pinned for this evaluation.

Without a complete candidate identity, capability tests and matched physical service tests cannot produce an admission result.

## Outcome

Return the runtime uplift as unselected and unadmitted.
Keep TT-Metal `0.74.0-rc6` as the Qwen3.8 production runtime.
Keep every current compatibility fallback.

This result does not reject a newer runtime on technical merit.
It closes this evaluation because no bounded candidate exists.

## What Changes

- Record the current runtime, model source, and compatibility fallbacks.
- Record that no immutable candidate closure was selected.
- Keep the repository runtime pins and `flake.lock` unchanged.
- Add no candidate package, service, capability, or speedup claim.
- Require a new validated Cairn change for a future named candidate.

## Impact

- **Nix**: no runtime input or lockfile changes.
- **Service**: the production Qwen3.8 closure remains unchanged.
- **Fallbacks**: sequential GDN and safe TP2 prefill remain selected.
- **Evidence**: the archive records the missing candidate identity.
- **Documentation**: no runtime-uplift performance claim is allowed.

## Out of Scope

- Selecting a runtime from public release age alone.
- Updating the repository-wide TT-Metal pin.
- Removing compatibility fallbacks.
- Inferring physical gains from simulator or version data.
