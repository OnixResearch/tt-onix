## Why

The B=4 candidate passed its typed physical gates in package `/nix/store/ibjk9b3lynppx1swbvi6x9qg07pdkpp8-qwen36-0.1.0`. The deployed Onix closure rebuilt Qwen with Python 3.14 as package `/nix/store/by2xl98jh01cjakf16zck4py9smbhmsr-qwen36-0.1.0`. The existing receipt cannot admit that different package identity.

## What Changes

- Restore `physically_admitted = false` for the B=4 profile.
- Keep the passing receipt as candidate-only evidence.
- Require an admission receipt to match the deployed package identity.
- Record the final deployed B=1 and B=4 smoke as non-admission evidence.

## Impact

- **Files**: Qwen batch profile, runtime specification, adoption documentation, and correction evidence.
- **Testing**: Deterministic profile export, package checks, Cairn gates, NixOS closure checks, and deployed health.
